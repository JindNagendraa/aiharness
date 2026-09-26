"""Mocked provider configuration, parsing, diagnostics, and fallback tests."""
import sys
import types
import io
import urllib.error
import pytest

from src import model
from src.model import ModelConfig, ModelError, ProviderConfig, TextModel
from src.providers import configured_providers
from src.providers.openrouter_provider import Provider as OpenRouterProvider
from src.providers.groq_provider import Provider as GroqProvider

ENV_NAMES=("AI_API_KEY","AI_MODEL","AI_BASE_URL","OPENROUTER_API_KEY","OPENROUTER_MODEL","GROQ_API_KEY","GROQ_MODEL","PRIMARY_PROVIDER","FALLBACK_PROVIDERS")

@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name,raising=False)

def test_missing_provider_credentials_are_clear(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL","configured-model")
    config=ProviderConfig.from_env("groq")
    assert config.api_key=="" and config.model=="configured-model"
    with pytest.raises(ModelError,match="GROQ_API_KEY"):
        TextModel("groq").generate([])

def test_missing_model_is_clear(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY","mock-key")
    with pytest.raises(ModelError,match="GROQ_MODEL"):
        TextModel("groq").generate([])

def test_openrouter_and_groq_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY","router-key")
    monkeypatch.setenv("OPENROUTER_MODEL","vendor/router-model")
    monkeypatch.setenv("GROQ_API_KEY","groq-key")
    monkeypatch.setenv("GROQ_MODEL","groq-model")
    router=ProviderConfig.from_env("openrouter")
    groq=ProviderConfig.from_env("groq")
    assert router.base_url=="https://openrouter.ai/api/v1"
    assert (router.api_key,router.model)==("router-key","vendor/router-model")
    assert (groq.api_key,groq.model)==("groq-key","groq-model")
    assert configured_providers()==["openrouter","groq"]

def test_evaluator_credentials_map_to_openrouter_compatibility(monkeypatch):
    monkeypatch.setenv("AI_API_KEY","evaluator-key")
    monkeypatch.setenv("AI_MODEL","evaluator-model")
    monkeypatch.setenv("AI_BASE_URL","https://evaluator.example/v1")
    config=ModelConfig.from_env()
    assert config.name=="openrouter"
    assert config.base_url=="https://evaluator.example/v1"
    assert TextModel()._order()==["openrouter"]

def test_openrouter_success_parses_chat_completion(monkeypatch):
    calls=[]
    class Response:
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"action":"finish"}'))]
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self,*args): return None
        def read(self): return b'{"choices":[{"message":{"content":"{\\"action\\":\\"finish\\"}"}}]}'
    def fake_urlopen(request,**kwargs):
        calls.append((request,kwargs))
        return FakeResponse()
    monkeypatch.setattr("urllib.request.urlopen",fake_urlopen)
    config=ProviderConfig("openrouter","mock-key","vendor/model","https://openrouter.ai/api/v1")
    result=OpenRouterProvider(config).generate("prompt",[{"role":"user","content":"hi"}],json_mode=True)
    assert result=='{"action":"finish"}'
    request,options=calls[0]
    assert request.full_url=="https://openrouter.ai/api/v1/chat/completions"
    assert options["timeout"]==120
    import json
    body=json.loads(request.data)
    assert body["model"]=="vendor/model"
    assert body["messages"]==[{"role":"user","content":"hi"}]
    assert body["tool_choice"]=="auto"
    assert body["parallel_tool_calls"] is False
    assert "response_format" not in body
    assert {tool["function"]["name"] for tool in body["tools"]}=={
        "list_files","read_file","search_text","search_names","write_file","run_command","git_status","finish"
    }

def test_openrouter_native_tool_call_becomes_agent_action_json(monkeypatch):
    calls=[]
    arguments={"path":"calculator.py","content":"def add(a, b): return a + b"}
    body={"model":"vendor/model","choices":[{"finish_reason":"tool_calls","message":{"content":None,"tool_calls":[{"type":"function","function":{"name":"write_file","arguments":__import__("json").dumps(arguments)}}]}}]}
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self,*args): return None
        def read(self): return json.dumps(body).encode()
    import json
    monkeypatch.setattr("urllib.request.urlopen",lambda request,**kwargs:(calls.append(request),FakeResponse())[1])
    config=ProviderConfig("openrouter","mock-key","vendor/model","https://openrouter.ai/api/v1")
    result=OpenRouterProvider(config).generate("prompt",[],json_mode=True)
    assert json.loads(result)=={"action":"write_file","arguments":{"path":"calculator.py","content":"def add(a, b): return a + b"}}

def test_openrouter_empty_response_reports_safe_metadata(monkeypatch):
    secret="router-secret-placeholder"
    body={"model":"vendor/model","usage":{"prompt_tokens":10,"completion_tokens":0},"choices":[{"finish_reason":"length","message":{"content":None}}]}
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self,*args): return None
        def read(self): return __import__("json").dumps(body).encode()
    monkeypatch.setattr("urllib.request.urlopen",lambda *args,**kwargs:FakeResponse())
    with pytest.raises(Exception) as caught:
        OpenRouterProvider(ProviderConfig("openrouter",secret,"vendor/model","https://openrouter.ai/api/v1")).generate("",[],json_mode=True)
    diagnostic=str(caught.value)
    assert "no content or valid action" in diagnostic
    assert "finish_reason=length" in diagnostic and "completion_tokens=0" in diagnostic
    assert secret not in diagnostic

def test_openrouter_uses_certifi_bundle_with_verification(monkeypatch):
    from src.providers import openrouter_provider
    import certifi
    import ssl
    calls={}
    original_create_context=ssl.create_default_context
    def fake_create_default_context(**kwargs):
        calls["ssl_kwargs"]=kwargs
        return original_create_context(**kwargs)
    monkeypatch.setattr(openrouter_provider.ssl,"create_default_context",fake_create_default_context)
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self,*args): return None
        def read(self): return b'{"choices":[{"message":{"content":"ok"}}]}'
    def fake_urlopen(request,**kwargs):
        calls["urlopen_kwargs"]=kwargs
        return FakeResponse()
    monkeypatch.setattr("urllib.request.urlopen",fake_urlopen)
    provider=OpenRouterProvider(ProviderConfig("openrouter","mock-key","model","https://openrouter.ai/api/v1"))
    assert provider.generate("",[])=="ok"
    context=calls["urlopen_kwargs"]["context"]
    assert calls["ssl_kwargs"]=={"cafile":certifi.where()}
    assert context.verify_mode==ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert calls["urlopen_kwargs"]["timeout"]==120

def test_openrouter_sanitizes_http_error(monkeypatch):
    secret="router-secret-placeholder"
    body=('{"error":{"type":"authentication_error","code":"invalid_api_key","message":"invalid '+secret+'"}}').encode()
    def fail(*args,**kwargs): raise urllib.error.HTTPError("https://router.invalid",401,"unauthorized",{},io.BytesIO(body))
    monkeypatch.setattr("urllib.request.urlopen",fail)
    with pytest.raises(Exception) as caught:
        OpenRouterProvider(ProviderConfig("openrouter",secret,"model","https://openrouter.ai/api/v1")).generate("",[])
    diagnostic=str(caught.value)
    assert "HTTP 401" in diagnostic and "invalid_api_key" in diagnostic
    assert secret not in diagnostic and "[REDACTED]" in diagnostic

def test_groq_success_and_json_mode(monkeypatch):
    observed={}
    class Response:
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"action":"finish"}'))]
    class Completions:
        def create(self,**kwargs): observed.update(kwargs); return Response()
    class Client:
        def __init__(self,**kwargs): observed["client"]=kwargs; self.chat=types.SimpleNamespace(completions=Completions())
    monkeypatch.setitem(sys.modules,"groq",types.SimpleNamespace(Groq=Client))
    result=GroqProvider(ProviderConfig("groq","mock-key","chosen-model")).generate("prompt",[{"role":"user","content":"hi"}],json_mode=True)
    assert result=='{"action":"finish"}'
    assert observed["client"]=={"api_key":"mock-key"}
    assert observed["model"]=="chosen-model"
    assert "response_format" not in observed
    assert observed["tool_choice"]=="auto"
    assert observed["parallel_tool_calls"] is False
    assert {tool["function"]["name"] for tool in observed["tools"]}=={
        "list_files","read_file","search_text","search_names","write_file","run_command","git_status","finish"
    }

def test_groq_native_tool_call_becomes_agent_action_json(monkeypatch):
    observed={}
    tool_call=types.SimpleNamespace(function=types.SimpleNamespace(name="write_file",arguments='{"path":"calculator.py","content":"def add(a, b):\\n    return a + b\\n"}'))
    class Response:
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None,tool_calls=[tool_call]))]
    class Completions:
        def create(self,**kwargs): observed.update(kwargs); return Response()
    class Client:
        def __init__(self,**kwargs): self.chat=types.SimpleNamespace(completions=Completions())
    monkeypatch.setitem(sys.modules,"groq",types.SimpleNamespace(Groq=Client))
    result=GroqProvider(ProviderConfig("groq","mock-key","openai/gpt-oss-120b")).generate("prompt",[],json_mode=True)
    import json
    action=json.loads(result)
    assert action=={"action":"write_file","arguments":{"path":"calculator.py","content":"def add(a, b):\n    return a + b\n"}}
    assert observed["tool_choice"]=="auto"
    assert "response_format" not in observed

def test_groq_json_only_mode_without_tools(monkeypatch):
    observed={}
    class Response:
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"action":"finish","arguments":{}}',tool_calls=None))]
    class Completions:
        def create(self,**kwargs): observed.update(kwargs); return Response()
    class Client:
        def __init__(self,**kwargs): self.chat=types.SimpleNamespace(completions=Completions())
    monkeypatch.setitem(sys.modules,"groq",types.SimpleNamespace(Groq=Client))
    result=GroqProvider(ProviderConfig("groq","mock-key","model")).generate("",[],json_mode=True,use_tools=False)
    assert result=='{"action":"finish","arguments":{}}'
    assert "tools" not in observed and "tool_choice" not in observed
    assert observed["response_format"]=={"type":"json_object"}

def test_groq_malformed_response_is_reported(monkeypatch):
    class Response: choices=[]
    class Completions:
        def create(self,**kwargs): return Response()
    class Client:
        def __init__(self,**kwargs): self.chat=types.SimpleNamespace(completions=Completions())
    # An incomplete response is turned into a sanitized provider diagnostic.
    monkeypatch.setitem(sys.modules,"groq",types.SimpleNamespace(Groq=Client))
    with pytest.raises(Exception,match="Groq failed"):
        GroqProvider(ProviderConfig("groq","mock-key","model")).generate("",[])

def test_groq_error_redacts_api_key(monkeypatch):
    secret="groq-secret-placeholder"
    class FakeError(Exception):
        status_code=429
        body={"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":f"key={secret}"}}
    class Completions:
        def create(self,**kwargs): raise FakeError("rate limited")
    class Client:
        def __init__(self,**kwargs): self.chat=types.SimpleNamespace(completions=Completions())
    monkeypatch.setitem(sys.modules,"groq",types.SimpleNamespace(Groq=Client))
    with pytest.raises(Exception) as caught:
        GroqProvider(ProviderConfig("groq",secret,"model")).generate("",[])
    assert "HTTP 429" in str(caught.value)
    assert secret not in str(caught.value) and "[REDACTED]" in str(caught.value)

def test_openrouter_to_groq_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY","router-key")
    monkeypatch.setenv("OPENROUTER_MODEL","router-model")
    monkeypatch.setenv("GROQ_API_KEY","groq-key")
    monkeypatch.setenv("GROQ_MODEL","groq-model")
    attempted=[]
    class FakeProvider:
        def __init__(self,config): self.name=config.name
        def generate(self,*args,**kwargs):
            attempted.append(self.name)
            if self.name=="openrouter": raise RuntimeError("temporary failure")
            return "groq result"
    monkeypatch.setattr(model,"create_provider",lambda config:FakeProvider(config))
    assert TextModel("openrouter",fallback=["groq"]).generate([],json_mode=True)=="groq result"
    assert attempted==["openrouter","groq"]

def test_groq_to_openrouter_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY","router-key")
    monkeypatch.setenv("OPENROUTER_MODEL","router-model")
    monkeypatch.setenv("GROQ_API_KEY","groq-key")
    monkeypatch.setenv("GROQ_MODEL","groq-model")
    attempted=[]
    class FakeProvider:
        def __init__(self,config): self.name=config.name
        def generate(self,*args,**kwargs):
            attempted.append(self.name)
            if self.name=="groq": raise RuntimeError("temporary failure")
            return "router result"
    monkeypatch.setattr(model,"create_provider",lambda config:FakeProvider(config))
    assert TextModel("groq",fallback=["openrouter"]).generate([])=="router result"
    assert attempted==["groq","openrouter"]
