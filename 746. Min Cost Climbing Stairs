class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # step 0 -> 0 cost
        # step 1 -> 0 cost
        # step 2 -> min(cost[0] + minCost(0), cost[1] + minCost(1))
        # step x -> min()
        # cost.length >= 2
        # associated step cost index -> xth step = cost[x]

        curr, prev = 0, 0
        for i in range(2, len(cost) + 1):
            temp = curr
            curr = min(curr + cost[i-1], prev + cost[i-2])
            prev = temp
        return curr
