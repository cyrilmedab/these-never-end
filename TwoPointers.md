# 125, Valid Palindrome

## Problem Statement
---
> Problem I: A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers. Given a string s, return true if it is a palindrome, or false otherwise.
---

## Thought Process

Don't bother with the whole reversing the string and comparing strategy; you'll need to iterate through the string 3 times and also require linear space. This is a simple two-pointer problem. The main challenge is remembering that alnum() and lower() exist, to check if something is alphanumeric and tow change upercase letters to lowercase. If you don't remember, maybe the interviewer will be fine with just assuming those functions exist, because they're just annoying to recreate, but not difficult. 

Anyways, classic two-pointers where you can progress the pointers on either end until you reach a valid alphanumeric character. If they're not equal, the string can't be a palindrome; if they are, you continue the while loop, incrementing the pointers further towards the center. You can exit the loop and return true when the left pointer is equal to or greater than the right pointer.

This runs with an O(N) time complexity and an O(1) space complexity.

```
class Solution:
    def isPalindrome(self, s: str) -> bool:
        front, back = 0, len(s) - 1

        while front < back:
            while not s[front].isalnum() and front < back:
                front += 1
            while not s[back].isalnum() and back > front:
                back -= 1

            if s[front].lower() != s[back].lower():
                return False
            front += 1
            back -= 1
        return True
```

# 167, Two Sum II - Input Array Is Sorted

## Problem Statement
---
> Problem I: Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length. Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2. The tests are generated such that there is exactly one solution. You may not use the same element twice. Your solution must use only constant extra space.
---

## Thought Process

This question has some weird stipulations about your output so make sure you read it carefully and fully process what is being asked. Also, the solution I've used is an optimization of the one's used by Neetcode and Leetcode, built using some of the comments on the post. This actually gives an O(logN) time complexity, making use of the fact that the array is already sorted. The other solutions have the same constant space complexity but a linear time complexity. 

Note that the binary search optimization only works for returning one solution, and we have to switch to using the linear solution if we wanted to pull multiple possible answers for the result. Not much else to explain, follows the standard binary sort approach, and increments like a two-pointer approach if binary isn't holding up.

```
class Solution:
    def binary_twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1

        while left < right:
            mid = left + (right - left) // 2
            curr_sum = numbers[left] + numbers[right]
            if curr_sum == target:
                return [left + 1, right + 1]
            elif curr_sum > target:
                right = mid if numbers[left] + numbers[mid] > target else right - 1
            else:
                left = mid + 1 if numbers[mid] + numbers[right] < target else left + 1
        return []
  
  # Leetcode O(N) time solution
  def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1

        while l < r:
            curSum = numbers[l] + numbers[r]

            if curSum > target:
                r -= 1
            elif curSum < target:
                l += 1
            else:
                return [l + 1, r + 1]
```

# 15, 16, 18 3Sum, 3Sum Closest, & 4Sum

## Problem Statement
---
> Problem I: Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0. Notice that the solution set must not contain duplicate triplets.
> 
> Problem II: Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
> 
> Problem III: Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that: 0 <= a, b, c, d < n
a, b, c, and d are distinct. nums[a] + nums[b] + nums[c] + nums[d] == target. You may return the answer in any order.
---

## Thought Process

A big family of problems. We're going to have to go through a lot of the offshoots and approaches, since if you get this problem, you're probably going to be asked any of a number of follow-ups given it's popularity. I lied, I'm just going to toss the optimal code for the rest below, too tired.

For the base 3Sum problem, all three approaches take O(N^2) time complexity and O(N) space, although the non-sorting solution (which I'm not going to explain too much) only provides the chance for slightly better space optimization depending on the input, even though it in actuality performs much slower due to the expensive set lookups. 

Anyways, we can solve it quickly by combining the linear TwoSum II solution with a for-loop. Sort the array so we have the elements in order and then set an outer for-loop that basically fixes the first value. the other two values can be determined by running TwoSumII on the rest of the array. TwoSumII will need slight modifications to append any matching pair it sees to the result array, but this is just a minor addition. You can also optimize the outer for-loop by breaking out when we reach positive integers; since the array is sorted, there's no way to get to zero including only positive integers. 

```
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:

        def two_sum(start: int, result: List[List[int]]) -> None:
            left = start + 1
            paired_sums = set()

            while left < len(nums):
                pair = -nums[start] - nums[left]
                if pair in paired_sums:
                    result.append([nums[start], pair, nums[left]])
                    while left + 1 < len(nums) and nums[left] == nums[left + 1]:
                        left += 1
                paired_sums.add(nums[left])
                left += 1



        nums.sort()
        result = []
        for i in range(len(nums) - 2):
            if nums[i] > 0: break
            if i > 0 and nums[i - 1] == nums[i]: continue
            two_sum(i, result)
        return result
        
   def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        dif = float('inf')

        for first in range(len(nums)):
            second, third = first + 1, len(nums) - 1
            while second < third:
                sum = nums[first] + nums[second] + nums[third]
                curr_dif = target - sum
                
                if abs(curr_dif) < abs(dif):
                    dif = curr_dif
                if sum < target:
                    second += 1
                else:
                    third -= 1
            if dif == 0: break
        
        return target - dif         
    
    # O(N^ (k-1) time complexity, where k is how many integers you want included in the sum. 
    # Linear space complexity worst case. Technically it's O(k) space complexity for the recursion stack
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def k_sum(nums: List[int], target: int, k: int) -> List[List[int]]:
            result = []
            if not nums: return result

            average = target // k

            if average < nums[0] or nums[-1] < average:
                return result
            if k == 2:
                return two_sum(nums, target)
            
            for i in range(len(nums)):
                if i == 0 or nums[i - 1] != nums[i]:
                    for subset in k_sum(nums[i + 1: ], target - nums[i], k - 1):
                        result.append([nums[i]] + subset)
            return result

        def two_sum(nums: List[int], target: int) -> List[List[int]]:
            result = []
            left, right = 0, len(nums) - 1

            while left < right:
                curr_sum = nums[left] + nums[right]

                if curr_sum < target or (left > 0 and nums[left] == nums[left - 1]):
                    left += 1
                elif curr_sum > target or (right < len(nums) - 1 and nums[right] == nums[right + 1]):
                    right -= 1
                else:
                    result.append([nums[left], nums[right]])
                    left += 1
                    right -= 1
            return result

        nums.sort()
        return k_sum(nums, target, 4)
```

# 11, 42 Container with the Most Water & Trapping Rain Water

## Problem Statement
---
> Problem I: You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]). Find two lines that together with the x-axis form a container, such that the container contains the most water. Return the maximum amount of water a container can store. Notice that you may not slant the container.
> 
> Problem II: Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
---

## Thought Process

The pictures make these way easier to follow, so look them up later; I'm lazy and procrastinating to my future self. These two problems are honestly very similar, with the main difference being that the other structures in the first problem aren't included to block the max volume; for the first problem, you only have two walls, and the second, you have multiple structures. 

For the first problem, we can set our pointers to either end. Then we want to figure out which of the pointers has the lower height. If we multiply that lower height by the distance between the pointers, then we have the max volume that can be contained between those two points. we can store that as the result for now. We then increment whichever pointer had the lower height closer towards the center, and then we repeat the process, comparing the max volume we calculate at each step with our stored max volumne, until the left pointer meets or passes the right pointer. That's it, super simple.

The second one should probably be a medium instead of a hard. It follows the same principle as the other problem, incrementing whichever pointer has the lower max height towards the center until they meet. At each step, we can subtract the current height at the pointer from the maximum height that the pointer we're looking at has seen so far and add that to our total. And again, that's it.

The reason this works is because we're trying to maximize how much water is contained. The amount of water that can be contained is determined by the max height of it's walls. We want to switch which pointer we're incrementing when one is greater than the other because each pointer doesn't know about the walls that the other pointer has encountered. So let's say we have the left pointer at a height of three, and the right pointer is 5 steps away at a height of 1. If we move the left pointer forward, we don't know if we'll encounter a wall higher that 3 by the time we reach the right pointer; maybe the limiting factor in this case is the height of 1 wall that the right pointer is on. Conversely, if we look at the right pointer, we can absolutely say that we'll encounter a wall higher than 1, because we know that we'll eventually hit the left pointers 3-height wall. 

Both of these, for their two-pointer solutions, run in linear O(N) time and a constant O(1) extra space. The trapping rain water problem has some other DP and stack solutions, but they both take O(N) space and honestly aren't as inuitive imo.
```
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_volume = 0

        while left < right:
            curr_volume = min(height[left], height[right]) * (right - left)
            max_volume = max(curr_volume, max_volume)

            if height[left] <= height[right]:
                left += 1
            else: 
                right -= 1
        return max_volume
    
    # Problem II
    def trap(self, height: List[int]) -> int:
        water_trapped = 0
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]
        
        while left < right:
            if left_max <= right_max:
                left += 1
                left_max = max(left_max, height[left])
                water_trapped += left_max - height[left]
            else:
                right -= 1
                right_max = max(right_max, height[right])
                water_trapped += right_max - height[right]

        return water_trapped
```











