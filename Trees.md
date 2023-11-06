# 226 Invert Binary Tree

## Problem Statement
---
> Problem I: Given the root of a binary tree, invert the tree, and return its root.
---

## Thought Process

The first tree problem of the Blind 75 and a relatively simple one that can be solved either iteratively or recursively with little effort. This would be a question where you'd want to clairfy as much as possible tho, because it's suspiciously easy and the interviewer might've hidden some random detail.

For example, clarify whether the tree is given in node representation, rather than as an array. Also confirm if you have to handle non-valid or empty trees. 

Recursive solution is super simple. You want to swap the two children node, which can be done in one line in Python without any dummy variables. Then the two children nodes are their own, smaller, binary trees, so you can call your function on both of the children nodes you just swapped. A helper isn't necessary for this, but I made one for clearer code and distintion.

The iterative approach is best done using a queue like a BFS (don't think you could do this with a stack easily). It's the same as the recursive approach where you only need to swap the children of the given node and then move on with the rest of the queue. After you pop the node out of the queue, the order for the rest of the loop doesn't matter: you can swap them first or add them to the queue first.

The worst-case time and space complexity for both methods are both O(N). However, the average-case time complexity for the recursive approach is O(H) or O(logN), due to a recursive call stack being the height of a tree. The iterative, BFS approach does not benefit from the properties of a tree, since the number of nodes at the bottom level is at most (N + 1) / 2, which in big O notation still simplifies to O(N).

```
class Solution:
    # Recursive Solution
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def invert_children(node: Optional[TreeNode]) -> Optional[TreeNode]:
            if not node: return node
            node.left, node.right = invert_children(node.right), invert_children(node.left)
            return node
        return invert_children(root)

    # Iterative, Queue Solution
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return root

        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
            node.left, node.right = node.right, node.left
        
        return root
```

# 104 Maximum Depth of Binary Tree

## Problem Statement
---
> Given the root of a binary tree, return its maximum depth.
> A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
---

## Thought Process

This can be kept brief because it's somewhat self evident. We can do this either recursively or iteratively using either a BFS or DFS; however, we should avoid the BFS since it's average -case space complexity is still O(N) while the other two have averages of O(logN).

The queue method is a good ol' level-order traversal, except with a variable to track each level. You can just return the level at the end, since that would be the maximum depth. The stack is a normal DFS traversal, except we're also storing the depth of each node alongside it in the stack. And big surprise, it's a standard recursion taking the max from the left and right children and adding 1 to it.

The time and space complexities for the BFS approach are both O(N). The time complexity for the stack (iterative and recursive) is O(N), and the space complexity is O(N) in the worst case and O(logN) in the average. 

```
class Solution:
    # Recursive
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def depth_finder(node: Optional[TreeNode]) -> int:
            if not node: return 0
            l_depth, r_depth = depth_finder(node.left), depth_finder(node.right)
            return 1 + max(l_depth, r_depth)
        return depth_finder(root)

    # Iterative, DFS
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        deepest, stack = 0, [[1, root]]
        while stack:
            depth, node = stack.pop()
            deepest = max(deepest, depth)
            if node.left: stack.append([depth + 1, node.left]) 
            if node.right: stack.append([depth + 1, node.right])
        return deepest

    # Iterative, BFS Level-Order
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        queue, depth = collections.deque([root]), 0
        while queue:
            length = len(queue)
            depth += 1

            for _ in range(length):
                node = queue.popleft()
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
        return depth
```

# 100, Same Tree

## Problem Statement
---
> Given the roots of two binary trees p and q, write a function to check if they are the same or not.
> Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
---

## Thought Process

Keeping it short, it's the same situation as usual. Iterative BFS or Recursive Stack. Same time complexities as normal for these approaches with a tree so I'm not going into detail on them this time.

The tricky case you have to handle is comparing nodes when one of them doesn't exist. Once you handle that, it's a simple case of iterating through both trees simultaneously. Check to make sure both nodes don't exist, which still satisfies them being the same, and then if only one of them doesn't exist, which would mean it's not the same.

```
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # Recursive - T: O(N), S: O(N) [avg. O(logN)]
        def recurse(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
            if not p and not q: return True
            if not p or not q: return False
            if p.val != q.val: return False

            return recurse(p.left, q.left) and recurse(p.right, q.right)
        
        return recurse(p, q)

        # Iterative, BFS - T: O(N), S: O(N) 
        def check_same(p : Optional[TreeNode], q: Optional[TreeNode]) -> bool:
            if not p and not q: return True
            if not p or not q: return False
            return p.val == q.val
        
        queue = collections.deque([(p, q)])
        while queue:
            p_node, q_node = queue.popleft()

            if not check_same(p_node, q_node): return False
            if not p_node: continue

            queue.append((p_node.left, q_node.left))
            queue.append((p_node.right, q_node.right))
        
        return True
```

# 572, Subtree of Another Tree

## Problem Statement
---
> Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.
> A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.
---

## Thought Process

skipping this for now cause I don't have the energy to go over hashing
```
ipsum
```

# 235, Lowest Common Ancestor of a Binary Search Tree

## Problem Statement
---
> Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.
> According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
---

## Thought Process

It's a binary SEARCH tree. Legit the most important detail, and then this becomse trivial. There's a recursive approach too, but the space complexity is objectively worse since the iterative approach works in O(1) space. 

All you do is go through the binary search tree node by node (starting at the root). Then you determine the point where only one of the target nodes is greater than the current node (and one is less than, obviously). If both are greater or lesser, that dictates which child of the current node that we want to check next.

The time complexity is O(N), since we might have to check every single node, but because this approach only allocates space for the dummy node variable (which you don't have to do if you use the input), it operates in O(1) space. 

```
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        node = root
        while node:
            if p.val > node.val and q.val > node.val: node = node.right
            elif p.val < node.val and q.val < node.val: node = node.left
            else: return node
```

# 1, Name

## Problem Statement
---
> 
---

## Thought Process

ipsum

```
ipsum
```

