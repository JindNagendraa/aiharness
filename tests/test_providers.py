"""Mocked provider configuration, parsing, diagnostics, and fallback tests."""
import json
import sys
import types
import io
import hmac
import os
import subprocess
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import pytest

from src import model
from src.model import ModelConfig, ModelError, ProviderConfig, TextModel
from src.providers import ProviderError, configured_providers
from src.providers.openrouter_provider import Provider as OpenRouterProvider
from src.providers.groq_provider import Provider as GroqProvider
from src.tui import run_tui

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

def test_ai_api_key_is_sent_as_openrouter_bearer_header_without_exposure(monkeypatch):
    secret = "SYNTHETIC_EVALUATOR_KEY_MUST_NOT_APPEAR_IN_OUTPUT"
    monkeypatch.setenv("AI_API_KEY", secret)
    monkeypatch.setenv("AI_MODEL", "test/endpoint-model")
    captured = {}

    class LocalOpenRouterMock(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["authorization"] = self.headers.get("Authorization")
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps({"choices": [{"message": {
                "content": '{"action":"list_files","arguments":{}}'
            }}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            # Keep even the local HTTP test server quiet; never log headers.
            pass

    server = HTTPServer(("127.0.0.1", 0), LocalOpenRouterMock)
    monkeypatch.setenv("AI_BASE_URL", f"http://127.0.0.1:{server.server_port}/api/v1")
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        text_model = TextModel()
        result = text_model.generate([{"role": "user", "content": "list files"}], json_mode=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

    header = captured.get("authorization")
    header_is_bearer = isinstance(header, str) and header.startswith("Bearer ")
    header_matches_configured_key = isinstance(header, str) and hmac.compare_digest(
        header, f"Bearer {secret}"
    )
    assert (header_is_bearer, header_matches_configured_key) == (True, True)
    assert result == '{"action":"list_files","arguments":{}}'
    assert secret not in result
    assert text_model.active_provider == "openrouter"

def test_tui_uses_same_ai_api_key_openrouter_provider_path(tmp_path, monkeypatch):
    secret = "SYNTHETIC_TUI_EVALUATOR_KEY_MUST_NOT_APPEAR_IN_OUTPUT"
    monkeypatch.setenv("AI_API_KEY", secret)
    monkeypatch.setenv("AI_MODEL", "test/endpoint-model")
    payload = {
        "model": "test/endpoint-model",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"tool_calls": [{"type": "function", "function": {
                "name": "finish", "arguments": '{"summary":"mock task complete"}'
            }}]},
        }],
    }
    observed = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(payload).encode()

    def fake_urlopen(request, **kwargs):
        header = request.get_header("Authorization")
        observed["bearer"] = isinstance(header, str) and header.startswith("Bearer ")
        observed["matches_key"] = isinstance(header, str) and hmac.compare_digest(
            header, f"Bearer {secret}"
        )
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tasks = iter(("finish a mocked task",))
    output = []

    def read_task(_prompt):
        try: return next(tasks)
        except StopIteration: raise EOFError

    run_tui(tmp_path, "true", input_fn=read_task, output_fn=output.append)

    assert observed == {"bearer": True, "matches_key": True}
    assert any("[SUCCESS]" in line for line in output)
    assert secret not in "\n".join(output)

def test_evaluator_key_only_does_not_require_an_invented_model(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "synthetic-evaluator-key")
    captured = {}
    class FakeProvider:
        actual_model = "endpoint-selected-model"
        def __init__(self, config): captured["config"] = config
        def generate(self, *args, **kwargs): return "mock completion"
    monkeypatch.setattr(model, "create_provider", FakeProvider)
    text_model = TextModel()
    assert text_model.generate([{"role": "user", "content": "hello"}]) == "mock completion"
    assert captured["config"].model == ""
    assert text_model.active_model == "endpoint-selected-model"

def test_openrouter_omits_unconfigured_model_from_evaluator_request(monkeypatch):
    import json
    monkeypatch.setenv("AI_API_KEY", "synthetic-evaluator-key")
    config = ProviderConfig.from_env("openrouter")
    requests = []
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"model":"endpoint/model-selected","choices":[{"message":{"content":"{\\"action\\":\\"list_files\\",\\"arguments\\":{}}"}}]}'
    def fake_urlopen(request, **kwargs):
        requests.append(json.loads(request.data))
        return FakeResponse()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenRouterProvider(config)
    assert json.loads(provider.generate("", [])) == {"action":"list_files","arguments":{}}
    assert "model" not in requests[0]
    assert provider.actual_model == "endpoint/model-selected"

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
    messages=[
        {"role":"system","content":"Return exactly one JSON object: {\"action\":\"...\",\"arguments\":{},\"done\":false}. Actions: list_files{}"},
        {"role":"user","content":"hi"},
    ]
    original_system=messages[0]["content"]
    result=OpenRouterProvider(config).generate("prompt",messages,json_mode=True)
    assert result=='{"action":"finish"}'
    request,options=calls[0]
    assert request.full_url=="https://openrouter.ai/api/v1/chat/completions"
    assert options["timeout"]==120
    import json
    body=json.loads(request.data)
    assert body["model"]=="vendor/model"
    assert body["messages"][0]["role"] == "system"
    assert "Return exactly one JSON object" not in body["messages"][0]["content"]
    assert "Call exactly one listed tool" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role":"user","content":"hi"}
    assert messages[0]["content"] == original_system
    assert body["tool_choice"]=="required"
    assert "Use the supplied native function tools" in body["messages"][0]["content"]
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
    assert "did not return a valid action/tool call" in diagnostic
    assert "finish_reason=length" in diagnostic and "completion_tokens=0" in diagnostic
    assert secret not in diagnostic

def test_openrouter_rejects_harmony_commentary_instead_of_treating_it_as_an_action(monkeypatch):
    secret = "SYNTHETIC_TEST_SECRET_DO_NOT_USE"
    content = f"<|start|>assistant<|channel|>commentary\nstatus: {secret}\n<|end|>"
    body = {"model":"openai/gpt-oss-120b","choices":[{"finish_reason":"stop","message":{"content":content}}]}
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(body).encode()
    requests=[]
    def fake_urlopen(request, **kwargs):
        requests.append(json.loads(request.data))
        return FakeResponse()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider=OpenRouterProvider(ProviderConfig("openrouter","mock-key","openai/gpt-oss-120b","https://openrouter.ai/api/v1"))
    with pytest.raises(Exception) as caught:
        provider.generate("", [{"role":"system","content":"Return an action"}], json_mode=True)
    diagnostic=str(caught.value)
    assert "did not return a valid action/tool call" in diagnostic
    assert "returned_model=openai/gpt-oss-120b" in diagnostic
    assert "finish_reason=stop" in diagnostic and "channel=commentary" in diagnostic
    assert secret not in diagnostic and content not in diagnostic
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["tools"]

def test_openrouter_harmony_failure_can_fall_back_to_groq(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-router-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "route/model")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq/model")
    attempted=[]
    class FakeProvider:
        def __init__(self, config): self.name=config.name
        def generate(self, *args, **kwargs):
            attempted.append(self.name)
            if self.name == "openrouter":
                raise ProviderError("OpenRouter did not return a valid action/tool call (channel=commentary)")
            return '{"action":"list_files","arguments":{}}'
    monkeypatch.setattr(model, "create_provider", lambda config: FakeProvider(config))
    generated=TextModel("openrouter", fallback=["groq"]).generate([], json_mode=True)
    assert json.loads(generated) == {"action":"list_files","arguments":{}}
    assert attempted == ["openrouter", "groq"]

def test_openrouter_diagnostic_redacts_secret_in_response_metadata(monkeypatch):
    secret="SYNTHETIC_TEST_SECRET_DO_NOT_USE"
    body={"model":secret,"choices":[{"finish_reason":"stop","message":{"content":"commentary"}}]}
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(body).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    provider=OpenRouterProvider(ProviderConfig("openrouter",secret,"route/model","https://openrouter.ai/api/v1"))
    with pytest.raises(Exception) as caught:
        provider.generate("", [], json_mode=True)
    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)

def test_openrouter_accepts_only_valid_structured_action_json_content(monkeypatch):
    body={"model":"route/model","choices":[{"finish_reason":"stop","message":{"content":"{\"action\":\"list_files\",\"arguments\":{}}"}}]}
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(body).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    provider=OpenRouterProvider(ProviderConfig("openrouter","mock-key","route/model","https://openrouter.ai/api/v1"))
    assert json.loads(provider.generate("", [], json_mode=True)) == {"action":"list_files","arguments":{}}

def test_openrouter_rejects_malformed_json_content_and_unknown_action(monkeypatch):
    bodies = (
        {"model":"route/model","choices":[{"finish_reason":"stop","message":{"content":"{bad json"}}]},
        {"model":"route/model","choices":[{"finish_reason":"stop","message":{"content":"{\"action\":\"run_shell\",\"arguments\":{}}"}}]},
    )
    for body in bodies:
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self): return json.dumps(body).encode()
        monkeypatch.setattr("urllib.request.urlopen", lambda *args, _response=FakeResponse(), **kwargs: _response)
        provider=OpenRouterProvider(ProviderConfig("openrouter","mock-key","route/model","https://openrouter.ai/api/v1"))
        with pytest.raises(Exception, match="did not return a valid action/tool call|unsupported action"):
            provider.generate("", [], json_mode=True)

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
        def read(self): return b'{"choices":[{"message":{"content":"{\\"action\\":\\"list_files\\",\\"arguments\\":{}}"}}]}'
    def fake_urlopen(request,**kwargs):
        calls["urlopen_kwargs"]=kwargs
        return FakeResponse()
    monkeypatch.setattr("urllib.request.urlopen",fake_urlopen)
    provider=OpenRouterProvider(ProviderConfig("openrouter","mock-key","model","https://openrouter.ai/api/v1"))
    assert json.loads(provider.generate("",[]))=={"action":"list_files","arguments":{}}
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

def test_explicit_empty_fallback_does_not_try_configured_groq(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "synthetic-evaluator-key")
    monkeypatch.setenv("AI_MODEL", "evaluator-model")
    monkeypatch.setenv("PRIMARY_PROVIDER", "openrouter")
    monkeypatch.setenv("FALLBACK_PROVIDERS", "")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    attempted = []

    class FakeProvider:
        def __init__(self, config): self.name = config.name
        def generate(self, *args, **kwargs):
            attempted.append(self.name)
            raise ProviderError(f"{self.name} failed: mock authentication failure")

    monkeypatch.setattr(model, "create_provider", lambda config: FakeProvider(config))
    with pytest.raises(ModelError, match="openrouter.*mock authentication failure"):
        TextModel().generate([], json_mode=True)
    assert attempted == ["openrouter"]

def test_explicit_empty_fallback_works_without_evaluator_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-router-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "router-model")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    monkeypatch.setenv("PRIMARY_PROVIDER", "openrouter")
    monkeypatch.setenv("FALLBACK_PROVIDERS", "")
    attempted = []

    class FakeProvider:
        def __init__(self, config): self.name = config.name
        def generate(self, *args, **kwargs):
            attempted.append(self.name)
            raise ProviderError("mock provider failure")

    monkeypatch.setattr(model, "create_provider", lambda config: FakeProvider(config))
    with pytest.raises(ModelError):
        TextModel().generate([], json_mode=True)
    assert attempted == ["openrouter"]

def test_omitted_fallback_preserves_configured_provider_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-router-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "router-model")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    monkeypatch.delenv("FALLBACK_PROVIDERS", raising=False)
    attempted = []

    class FakeProvider:
        def __init__(self, config): self.name = config.name
        def generate(self, *args, **kwargs):
            attempted.append(self.name)
            if self.name == "openrouter":
                raise ProviderError("mock primary failure")
            return "groq fallback result"

    monkeypatch.setattr(model, "create_provider", lambda config: FakeProvider(config))
    assert TextModel("openrouter").generate([]) == "groq fallback result"
    assert attempted == ["openrouter", "groq"]

def test_evaluator_key_intentionally_selects_openrouter(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "synthetic-evaluator-key")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    monkeypatch.setenv("PRIMARY_PROVIDER", "groq")
    monkeypatch.setenv("FALLBACK_PROVIDERS", "")
    assert TextModel()._order() == ["openrouter"]

def test_make_recipe_preserves_inline_provider_environment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "read_provider_env.py"
    probe.write_text(
        "import os\n"
        "print(os.environ.get('PRIMARY_PROVIDER', '') + '|' + "
        "('empty' if 'FALLBACK_PROVIDERS' in os.environ and not os.environ['FALLBACK_PROVIDERS'] else 'other'))\n"
    )
    extra_makefile = tmp_path / "env-probe.mk"
    extra_makefile.write_text(f"provider-env-probe:\n\t@$(BIN)/python {probe}\n")
    env = os.environ.copy()
    env["PRIMARY_PROVIDER"] = "openrouter"
    env["FALLBACK_PROVIDERS"] = ""
    output = subprocess.run(
        ["make", "-f", str(root / "Makefile"), "-f", str(extra_makefile), "provider-env-probe"],
        cwd=root, env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert output == "openrouter|empty"

def test_provider_failure_diagnostic_names_all_attempts(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-router-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "router-model")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "groq-model")
    monkeypatch.setenv("FALLBACK_PROVIDERS", "groq")
    attempted = []

    class FakeProvider:
        def __init__(self, config): self.name = config.name
        def generate(self, *args, **kwargs):
            attempted.append(self.name)
            if self.name == "openrouter":
                raise ProviderError("OpenRouter failed: HTTP 401 authentication error")
            raise ProviderError("Groq failed: HTTP 413 request too large")

    monkeypatch.setattr(model, "create_provider", lambda config: FakeProvider(config))
    text_model = TextModel("openrouter")
    with pytest.raises(ModelError) as caught:
        text_model.generate([], json_mode=True)
    message = str(caught.value)
    assert attempted == ["openrouter", "groq"]
    assert "OpenRouter failed: HTTP 401" in message
    assert "Groq failed: HTTP 413" in message
    assert text_model.active_provider == ""

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
