# 746 Min Cost Climbing Stairs

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
