# 416 Partition Equal Subset Sum

## Problem Statement
---
> Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.
---

## Thought Process

This is a fun one that can be solved in two different ways that are both equally viable and important to remember. Both solutions have significant tradeoffs between space complexity and time complexity, so it's critical to discuss that aspect with an interviewer and decide on the method based on their responses. For example, the 2nd method with the bottoms-up, single DP array isn't feasible if the integers can be negative, so you'd have to go with the recursive top-down. Or perhaps we don't care about the worst case time complexity and are willing to accept that risk in exchange for faster average performance. Basically, even thoough the time complexities for both approaches are the same, the recursive DFS solution performs better than the bottoms-up solution, and the bottoms-up solution reduces the space complexity from O(NM) to O(M).

Both solutions start the same way. We want to determine what the partition sum would be. We take the total sum of the array. Now, if the total sum is an odd number, then the problem is not possible and we can return false immediately. This is because an odd number cannot be divided into two of the same number (kinda like a defining feature of odd numbers, dumbass). If it's even though, we divide it by 2 and that's our target value. Here's where we can have different approaches.

The first solution is a standard recursive, post-order DFS. We're essentially choosing at each number in the input array whether or not we want to include this in our partitioned array, and we compute both paths by calling a recursive function. The benefit of this approach is that we can have it terminate immediately once we find a working partition, wheras the next solution still methodically works through a range and gives a more consistent time. The really cool addition that makes this so much faster is that we can sort in decreasing order the input nums array before we start the recursion. Why? If we start with the larger values, we're more likely to hit sooner partition sums that exceed our target sum, so we can abort the branch sooner. This minimizes the amount of recursion levels we actually have to go through. Finally, we want to have a cache so that we're no recalculating the same operations. We can use either a plain ol' dictionary or Python's LRU cache (@lru_cache()). 

The time complexity is O(MN) in the worstcase where there are no overlapping calls and use of the cache, where M is the target sum and N is the number of elements in the input array. The space complexity is also O(MN) for allocating the 2D array memo for the recursive stack.

The second solution operates using a 1D boolean array the length of the target sum + 1. This means the indices range from 0 to the target, inclusive. We can then initiate the 0-index to be true. Now we can do a nested for-loop and for every number in our input array, we'll loop through all the values from our target sum down to our current number (a slight optimization from doing target sum down to 0). If the index at that position is already true, it stays true, otherwise it's set to be the same value as the current index minus the current num (so the index at i-num). This is essentially saying, hey, we could have made a partition that equals this previous sum, and now we're just adding this num to that sum, so obviously we can make it too. Intuitive once you see it, but draawing it out might help if I get around to it. 

The time complexity is O(MN) because we're iteratively filling an M-length array N times, and the space complexity is O(M), for the aforementioned M-length array.

```
class Solution:
    # THe recursive method
    def canPartition(self, nums: List[int]) -> bool:

        @lru_cache(maxsize=None)
        def dfs(ind: int, total: int) -> bool:
            if total == 0: return True
            if ind == len(nums) or total < 0: return False
            return dfs(ind + 1, total - nums[ind]) or dfs(ind + 1, total)
        
        total = sum(nums)
        if total % 2: return False

        nums.sort(reverse=True)
        target = total // 2
        return dfs(0, target)

# The iterative 1D bottoms-up DP solution
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2: return False

        target = total // 2
        partition_sums = [False] * (target + 1)
        partition_sums[0] = True

        nums.sort(reverse = True)
        for num in nums:
            for i in range(target, num - 1, -1):
                partition_sums[i] = partition_sums[i] or partition_sums[i - num]
                if partition_sums[target]: return True
        return False
```
