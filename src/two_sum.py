"""LeetCode style two sum problem.

Provides a function `two_sum(nums, target)` that returns a list of two indices
such that `nums[i] + nums[j] == target`. It assumes exactly one solution
exists and does not use the same element twice.
"""

def two_sum(nums, target):
    """Return indices of the two numbers that add up to `target`.

    Args:
        nums (list[int]): List of integers.
        target (int): The target sum.

    Returns:
        list[int]: A list containing the two indices.
    """
    # Use a dict to store value -> index.
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    # If no solution found, raise an error.
    raise ValueError("No two sum solution")
