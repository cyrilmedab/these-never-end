# 70 & 746 Climbing Stairs and Min Cost Climbing Stairs

## Problem Statement
---
> You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps. You can either start from the step with index 0, or the step with index 1. Return the minimum cost to reach the top of the floor.
---

## Thought Process
This is essentially the same problem as Leetcode #70 Climbing Stairs, except with a cost associated to moving along the steps. Also notably, the goal in this problem isn't to just get to the top-most step, but to get to the **landing** beyond the top step, thereby getting off the stairs. This is an important difference that affects the for-loop that we run, but only means we add an extra iteration. Otherwise, the code is relatively unchanged from the simplified version. 

This time, instead of adding the previous two steps together, we want to know which of them is cheaper overall to get to the top. The minimum cost to leave from a step can be described as the minimum cost to get there plus the cost of the step itself. Since you can only reach step N from either step N-1 or N-2, the minimum cost to get to step N will be the lower value of the cost to leave step N-1 or N-2. Our initial values are also changed slightly, with the initial cost to leave being zero, as opposed to the distinct way to leave being 1.

This can also be done recursively, but my implementation here gives us an O(1) space complexity. The time complexity in both cases is O(N).
THe first function is the solution to the simplified version, and the second is the solution to this problem.

```
class Solution:
    def climbStairs(self, n: int) -> int:
        curr, prev = 1, 0
        for i in range(n):
            temp = curr
            curr = prev + curr
            prev = temp
        return curr
        
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        curr, prev = 0, 0
        for i in range(2, len(cost) + 1):
            temp = curr
            curr = min(curr + cost[i-1], prev + cost[i-2])
            prev = temp
        return curr
```

# 198 & 213, House Robber I and II

## Problem Statements
---
> Problem I: You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if **two adjacent houses** were broken into on the same night. Given an **integer array nums** representing the amount of money of each house, return the **maximum amount of money** you can rob tonight **without alerting the police**.
> 
> Problem II: You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are **arranged in a circle**. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night. Given an **integer array nums** representing the amount of money of each house, return the **maximum amount of money** you can rob tonight **without alerting the police**.
---

## Thought Process
These two problems are identical except that the second is a continuous loop, meaning that the first and last houses are adjacent. This seems like a complicated change but has an extremely simple solution. I wrote nearly the exact same code as Neetcode, except in House Robber II; I didn't refactor my code like I should have. I'm adding both tho for completeness. 

Luckily, these also follow a similar concept to the stairs problems above. Take the first step, a trivial case; the max value is going to be the value of the house. Now, what if there are two houses? If the second house's value is greater than the first house's, the we want to select the second house as the primary option. Adding a third house is where we see the formula more clearly. For the third house, we can either skip it and only visit the second house, or we can visit the first house and then the third house. We want the maximum of this decision. Therefore, the optimal path for **x<sup>th</sup>** house is the max between **(the optimal cost for x-1<sup>th</sup>)** house and **(The cost of x<sup>th</sup> + the optimal cost of x-2<sup>th</sup>)** houses. And this will continue to hold up for all the houses we visit, as long as we don't connect the first and last house. So Problem 2 now!

This is actually surprsingly simple if you think about it. There are three outcomes possible: Either we take from the first house, the last house, or neither. Tackling the first outcome, if we just exclude the last house, then it just becomes the House Robber I problem again! The same is true for the second outcome; if we exclude the first house, we run House Robber I again. Now how do we account for when we don't include either? We already have! Our solution will naturally exclude whichever house we included if it's not part of the optimal scheme, and the other was already excluded manually. So we can just return the max value between these two outcomes. Both of these solutions have O(N) time complexity and O(1) space complexity.

```
class Solution:
    #Robber I
    def rob(self, nums: List[int]) -> int:
        prev_max, curr_max = 0, 0
        for num in nums:
            temp = curr_max
            curr_max = max(curr_max, prev_max + num)
            prev_max = temp
        return curr_max
        
    #Robber II 
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        prev1, curr1, prev2, curr2 = 0, 0, 0, 0

        for i in range(len(nums) - 1):
            temp = curr1
            curr1 = max(curr1, prev1 + nums[i])
            prev1 = temp
        
            temp = curr2
            curr2 = max(curr2, prev2 + nums[i+1])
            prev2 = temp
        return max(curr1, curr2)
        
    # Neetcode's Robber II solution
    def rob(self, nums: List[int]) -> int:
        return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))

    def helper(self, nums):
        rob1, rob2 = 0, 0

        for n in nums:
            newRob = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = newRob
        return rob2
```

# 5 & 647, Longest Palindromic Substrings and Palindromic Substrings

## Problem Statements
---
> Problem I: Give a string s, return *the longest* palindromic substring.
> 
> Problem II: Given a string s, return _the number of **palindromic substrings** in it_.
---

## Thought Process
Right off the bat, we should clarify that palindromes are strings that reads the same when backwards as when forwards. Substrings are _continuous and unbroken_ sets of strings within a main string. Note that a substring isn't required to be smaller than the main string. For example, a substring of "Cheese" could not only be "hee" but also "Cheese".

Another important thing to clarify for Problem 1 is how we should handle palindromic substrings of the same length. Are both valid, or should we be prioritizing the first one we encountered when going from left to right, or the last one? My solution assumes that both are valid and it doesn't matter which one you return, but this can be adjusted for any of the cases by modifying the IF-statement in the FOR-loop in longestPalindrom.


```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def palindrome_length_check(s: str, left: int, right: int) -> int:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        start_ind, end_ind = 0, 0
        for i in range(len(s)):
            odd_length = palindrome_length_check(s, i, i)
            even_length = palindrome_length_check(s, i, i + 1)
            
            largest_length = max(odd_length, even_length)
            if largest_length >  end_ind - start_ind:
                start_ind = i - (largest_length - 1) // 2   # These index calcs were a bit tricky to fully comprehend, at least for me
                end_ind = i + 1 + largest_length // 2       # The 1 is added because slicing the string doesn't include the last index, so we go one beyond
        return s[start: end]

```




