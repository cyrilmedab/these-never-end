# 20, Valid Parentheses

## Problem Statement
---
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
---

## Thought Process

This is apparently the only "pure" stack problem in the Blind 75 (at least according to Neetcode's categories), but understanding it does give you a firm grasp on how stacks can and will be used. I don't believe this problem on its own  needs too much of an explanation tho; if you're familiar with stacks, the solution should come to you pretty quickly.

For valid parenthesis, each closing bracket has a paired opening bracket. You can use this to make a dictionary for a constant time lookup of the pairs. The creation of this dictionary will still take constant space too, since there will only be three pairs not matter how long the input string is.

From here, we can loop through each character in the string. If the character is an open bracket, we can append it to the stack and move on with our lives. If the character is a closing bracket, we can compare its paired value from the dictionary to the last bracket in our stack. If there's a match, we pop the value off the stack in constant time. If there's not, we can stop everything and return False.

The final return value is even simpler. If the stack still exists (aka it's not empty), then we can't have a valid string of parenthesis since that means that there are open brackets that were never closed (and therefore popped off the stack). If the stack is empty tho, we've successfully closed all the brackets and can confirm that the given string is a valid sequence of parentheses.

I'll include both mine and Neetcode's code for this one, because he had a slightly different coding style which some might find a bit neater (or should I say, neeter! haha kill me). 

```
class Solution:
    def neetcode_isValid(self, s: str) -> bool:
        Map = {")": "(", "]": "[", "}": "{"}
        stack = []

        for c in s:
            if c not in Map:
                stack.append(c)
                continue
            if not stack or stack[-1] != Map[c]:
                return False
            stack.pop()

        return not stack
        
    # function name is what I say every morning when I look in the mirror
    def me_isValid(self, s: str) -> bool:
      closing_pairs = {")": "(", "]": "[", "}": "{"}
      stack = []
      
      for char in s:
        if char in closing_pairs:
          last_bracket = stack.pop() if stack else '' # empty string for comparison's sake
          if closing_pairs[char] != last_bracket:
            return False
        else:
          stack.append(char)
      
      return not stack
```
