# 217, 219 Contains Duplicate I & II

## Problem Statement
---
> Problem I: Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
>
> Problem II: Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
---

## Thought Process

217 is the one that's included in the Blind 75, not 219. However, I feel that 217 is so aggressively easy; the solution feels like it's literally to remember that sets exist. My suspicion is that's actually part of the challenge when this one is brought up in coding interviews, cause Leetcode suggests that it's used surprisingly often.

The problem is so generic that it could be worded in different contexts, given different meanings, have a lot of filler information--all to obscure the fact that the solution is incredibly simple. So what the interviewer would be testing is your ability to parse information and isolate the important bits. 

Anyways, cause it's too simple, I included the next stage of the problem, which while still being an easy problem, has a slightly more fun twist.

Reiterating, the solution to the first problem is to use a set. Sets give us constant time lookup and will only store a distinct value once; our set can't contain the same value twice. Working through the given array then, we can add each value in the array to the set. Each time we add a value though, we also want to check if the value is already in the set. If it's in the set, we satisfy our goal of detecting if a number appears **at least twice** and can return true. We don't care about what number it is or which index we found it in or anything else.

The next problem asks us a similar question, except this time, we can see that we have to keep track of the indices as well. Sets aren't up to the task, so we can use the heavier hitter: dictionaries. We can use the integers in the given input array as our dictionary keys, and store the index where we found them in their value. 

Once again, we can loop through the input array, storing the index of each integer we see in our dictionary. At this point, we can use our dictionary to look up a specific value and the index where we saw it. For each number, we also want to check if it satisfies our condition! If the number is already in the dictionary (as a part of the keys), then we can subtract the stored index of our current value from the current index of our current value. If that's <= k, then we can return true.

An important thing to note is that we don't even have to compute the absolute value of the difference. Since we're progressing from left to right, we know that our current index will always be greater than our stored index. Therefore, the absolute value calculation is unecessary if we put current index - stored index.

Another question is if we have more than just two copies of a value in the input array. Do we need to store all the indices and check our wincon against all of them each time we find a new repeated value? No. We will only care about the most recently seen index of the current number. Any prior appearances would end up having a greater difference in the index values, so are unnecessary to check against.

All that said, I'm an idiot and Leetcode Premium shows that you can solve the second problem even more efficiently with a smaller space complexity using just a set. You do this by combining a set with a sliding window approach. The set will only contain the k most recent elements; we can add each elemnt we see to the set and remove the oldest value (aka the value of num[curr_index - k]) from the set. This gives us the same linear time complexity of the dictionary approach but a better space complexity. Instead of O(N), we get O(min(N, k)). This is because our set will require k space but never more than N space.

Note that while the time complexity is technically the same, leetcode submission show that the dictionary approach regularly runs faster. Not importan for interviews, just cool.

```
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
        
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        seen_indices = {}

        for i in range(len(nums)):
            if nums[i] in seen_indices:
                if i - seen_indices[nums[i]] <= k:
                    return True
            seen_indices[nums[i]] = i
        return False
        
   # The set approach
   def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        seen_indices = set()

        for i in range(len(nums)):
            if nums[i] in seen_indices:
                return True
            seen_indices.add(nums[i])
            if len(seen_indices) > k:
                seen_indices.remove(nums[i - k])
        return False
```

# 242, Valid Anagram

## Problem Statement
---
> Problem I: Given two strings s and t, return true if t is an anagram of s, and false otherwise. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
---

## Thought Process
