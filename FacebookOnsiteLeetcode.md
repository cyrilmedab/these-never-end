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

# 1123 Lowest Common Ancestors of Deepest Leaves

## Problem Statement
---
> Given the root of a binary tree, return the lowest common ancestor of its deepest leaves.
> Recall that:
>     The node of a binary tree is a leaf if and only if it has no children
>     The depth of the root of the tree is 0. if the depth of a node is d, the depth of each of its children is d + 1.
>     The lowest common ancestor of a set S of nodes, is the node A with the largest depth such that every node in S is in the subtree with root A.
---

## Thought Process

Two paths we can go down, the recursive one is shorter and more performant but the iterative one is the one that I thought of when I first saw this problem. Going to keep it short and sweet.

The iterative approach is to do a level order, BFS traversal of the tree using a queue. At the start of each level, we store the first and last value, or the leftmost and rightmost nodes. When the queue is empty, that means we were just on the deepest level and have the furthest apart nodes on that level. From there, we can either recursively or iteratively call a helper function to find the LCA of the two nodes, should be trivial at this point. I'm going to list both helper methods below.

The time complexity of this approach is O(N) and a space complexity of O(N) as well. This is because we'll be hitting every node in the level traversal and worst-case storing every node in the queue. The LCA helper would also require O(N) time and space, for both the iterative and recursive solutions. The downside to this approach, depsite being so intuitive, is that it requires at least 2 passes through the tree, and in the worst case 3.

The recursive approach is a simple DFS, using a helper that returns a tuple containing the maximum depth of a given path and the LCA of that path. The LCA is determined/updated when the left and right paths both have the same max depth. It's a simple solution with O(N) time and space, and it only needs one-pass through the tree. Looking at the code below (it's te first one listed), we can easily see why this is the preferred method, between being much shorter to write and more performant as well.

```
class Solution:
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def deepest_recurse_lca(node: Optional[TreeNode]) -> tuple[int, Optional[TreeNode]]:
            if not node: return 0, None
            left_depth, left_lca = recursive_lca(node.left)
            right_depth, right_lca = recursive_lca(node.right)

            if left_depth > right_depth: return left_depth + 1, left_lca
            elif right_depth > left_depth: return right_depth + 1, right_lca
            else: return left_depth + 1, node
        return deepest_recurse_lca(root)[1]

        def iterative_lca(node1: Optional[TreeNode], node2: Optional[TreeNode]) -> Optional[TreeNode]:
            parents, queue = {root : None}, deque([root])            
            while node1 not in parents or node2 not in parents:
                node = queue.popleft()
                if node.left:
                    parents[node.left] = node
                    queue.append(node.left)
                if node.right:
                    parents[node.right] = node
                    queue.append(node.right)
            
            parent1, ancestors1 = node1, set()
            while parent1:
                ancestors1.add(parent1)
                parent1 = parents[parent1]
            
            lca = node2
            while lca not in ancestors1: lca = parents[lca]
            return lca

        def recursive_lca(node: Optional[TreeNode], target1: Optional[TreeNode], target2: Optional[TreeNode]) -> Optional[TreeNode]:     
            if not node or node == target1 or node == target2: return node
            left = recursive_lca(node.left, target1, target2)
            right = recursive_lca(node.right, target1, target2)

            if left and right: return node
            else: return left or right
        
        queue = deque([root])
        while queue:
            leftmost, rightmost = queue[0], queue[-1]
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
        return iterative_lca(leftmost, rightmost)
        return recursive_lca(root, leftmost, rightmost)
```

# 438 Find All Anagrams in A String

## Problem Statement
---
> Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.
> An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
---

## Thought Process
It's exactly like all other Anagram problems. This one only uses lowercase English letters, so you can use an array to track the frequencies of characters instead of a dictionary, to reduce the complexity the hashing always brings. Get the count of both strings for length p, and then start iterating through the array and appending the start index when the counts match at 26. Code explains it better since there's really nothing new about this problem.

```
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_size, s_size = len(p), len(s)
        start_indices = []
        if p_size > s_size: return start_indices

        p_count, s_count = [0] * 26, [0] * 26
        for i in range(p_size):
            p_ind, s_ind = ord(p[i]) - ord('a'), ord(s[i]) - ord('a')
            p_count[p_ind] += 1
            s_count[s_ind] += 1
        
        matched = 0
        for i in range(len(p_count)): 
            if p_count[i] == s_count[i]: matched += 1
        if matched == 26: start_indices.append(0)

        for i in range(1, s_size - p_size + 1):
            remove_ind, add_ind = ord(s[i-1]) - ord('a'), ord(s[i + p_size - 1]) - ord('a')

            s_count[remove_ind] -= 1
            if s_count[remove_ind] == p_count[remove_ind]: matched += 1
            elif s_count[remove_ind] == p_count[remove_ind] - 1: matched -= 1

            s_count[add_ind] += 1
            if s_count[add_ind] == p_count[add_ind]: matched += 1
            elif s_count[add_ind] == p_count[add_ind] + 1: matched -= 1

            if matched == 26: start_indices.append(i)
        
        return start_indices
```
