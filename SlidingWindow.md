# 121, 122 Best Time to Buy and Sell Stock I & II

## Problem Statement
---
> Problem I: You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
>
> Problem II: You are given an integer array prices where prices[i] is the price of a given stock on the ith day. On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day. Find and return the maximum profit you can achieve.
---

## Thought Process

I added problem II because it's a not too difficult addition to the solution you find for the first problem. Both of these run in linear O(N) time with a single pass, and O(1) constant space.

The first problem, you just need a variable to hold the max profit as you loop through the array, updating it as you go. When do you update it? Well, we want to maximize how much money we can earn buy buying on only 1 day and selling on only 1 day. If we start as if we're buy on thevery first day, the moment that we find a price that is lower than what we would have bought at, we want to switch our buying day to that new lower price. Once we switch, we've already calculated the max profit we could achieve from purchasing stock on our previous choce. We can keep this algorithm going until we reach the end of the price list, and then we can return the profit. The code might be cleaner than this explanation tho.

The second problem allows us to buy and sell on multiple days. We can actually solve the second problem by taking our solution to the first problem and adding an extra variable to store the current max profit for an interval. There's actually a more optimal and even easier solution however.

As we iterate through the aray of prices, if our current day was a price increase over the previous day, we can add the difference to our total profit. On days where the price is actually less than our previous day, we can pretend that we already sold our stock and not add anything. The below picture makes this very clear.
![image](https://user-images.githubusercontent.com/118993796/224596612-8990dcfa-ba72-4a7e-bd9a-9860b1d149e2.png)

```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit, buy_value = 0, prices[0]

        for sell_value in prices:
            if sell_value < buy_value:
               buy_value = sell_value
            interval_profit = sell_value - buy_value
            max_profit = max(max_profit, interval_profit)
        
        return max_profit

class SolutionII:
    def maxProfitII(self, prices: List[int]) -> int:
        total_profit = 0

        for sell_day in range(1, len(prices)):
            if prices[sell_day] <= prices[sell_day - 1]: continue
            total_profit += prices[sell_day] - prices[sell_day - 1]

        return total_profit
```

# 3, Longest Substring Without Repeating Characters

## Problem Statement
---
> Problem I: Given a string s, find the length of the longest substring without repeating characters.
---

## Thought Process

This one's a bit tricky tbh. My first thought was to use a queue, but that doesn't give us the constant lookup time we'd need for elements in the middle of the queue. The more obvious chose is a set, and using that to identify if we're seeing a repeated character, but that doesn't keep track of the order in which we add characters. We can remedy this with a sliding window. Surprise!

If we have a pointer set to the start of the input string, this can act as our pointer for the start of the substring as well. Each time we see a character, we can add it to the set. If it's already in the set tho, we'll enter a loop. Now we can use our pointer to check what character is at the start of the string. We can then remove that chracter from the set in constant time and then increment our pointer by 1. So our pointer is still pointing to the start of the substring. We continue this loop, removing characters until the character we're currently on in the main loop is no longer contained in the set. 

The benefit of this approach is that we can then have a cosntantly updating max_length value that compares itself against the distance between where we're at and the starting pointerr, since that would give us the length of the substring. This gives us an O(N) time complexity and an O(min(N, M)) space complexity, where n is the size of the string and m is the size of the charset/ number of unique elements that could possibly show up. 

Leetcode actually has an even further optimized sliding window solution. While this new approach has the same O(N) time complexity, the new approach will require only a single pass through the string, wheras the previous approach would visit each character twice in the worst case.

If we use a dictionary instead of a set, we can store the indices at where a character appears. So instead of sending our pointer through elements we've already visited until we get to the repeating character, we can simply jump our pointer up to the appropriate index. This would still have the same space complexity as well, even though dictionaries take up more actual space than a set.

```
class Solution:
    # Non-optimal O(2N) sliding window solution
    def lengthOfLongestSubstring(self, s: str) -> int:
        substring_start = max_length = 0
        chars = set()


        for ind in range(len(s)):
            while s[ind] in chars:
                chars.remove(s[substring_start])
                substring_start += 1

            chars.add(s[ind])
            max_length = max(max_length, ind - substring_start + 1)

        return max_length
   
   # Optimal O(N) sliding window solution
   def lengthOfLongestSubstring(self, s: str) -> int:
        substring_start = max_length = 0
        chars = {}

        for ind in range(len(s)):
            if s[ind] in chars:
                substring_start = max(chars[s[ind]], substring_start)

            chars[s[ind]] = ind + 1
            max_length = max(max_length, ind - substring_start + 1)
                
        return max_length
```

# 424, Longest Repeating Character Replacement

## Problem Statement
---
> Problem I: You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times. Return the length of the longest substring containing the same letter you can get after performing the above operations.
---

## Thought Process

Apparently we can use a pure binary search to reduce (kinda) the time complexity to O(NlogN) (actually this is a bit of a later move, depending on the size of M, it could be equal to N). However, the big bucks is if we combining some of the concepts that the binary search approach is based on with the sliding window to get a linear runtime O(N).

Basically, this relies on the principle that if a valid window of size X exists, then valid windows of size X - 1, X - 2, etc. exist. While that sounds basic, that really means that if we find a valid window of size X, we don't need to send our start-of-substring pointer all the way up until the substring is valid again. We can just start incrementing the start pointer alongside the end pointer as well when the substring isn't valid. This means that when the end pointer hits the end of the array, we're done. So that explains why we can achieve this in linear time. When or if our substring becomes valid again (we'll still be maintaining our frequency counter) we'll stop incrementing our starting pointer, thereby increasing the window while the substring is valid. 

A big benefit of the never decreasing window size is that we don't even need to keep track of the max_length anymore! This is huge, since we can avoid the costly max() comparisons, and it greatly improves performance if not compleity. Once we finish the loop, the start pointer still exists, and we know that our end pointer will have had to have been at the end of the list. Therefore, we can substract the location of the start pointer from the length of the array just once! All done while returning the result.

Note: Really just read the leetcode editorial for this one. It's actually really fucking good for once, and has insights on binary search that I'm sure you've already forgotten, you dingbat.

Yeah the explanation is rough and the explanation is actually good. I'm going to toss the optimal linear time complexity code down there, and if you don't understand it, look at the editorial in Leetcode. I'm not even going to watch the neetcode video on this one, cause the explanation was good enough. 

```
class Solution:
    # Optimal O(N) time solution, using the never-shrinking window appraocj
    def characterReplacement(self, s: str, k: int) -> int:
        freq_counter = collections.defaultdict(int)
        max_freq = max_length = start = 0

        for ind in range(len(s)):
            freq_counter[s[ind]] += 1
            max_freq = max(max_freq, freq_counter[s[ind]])

            # the length of the current substring - the current most frequenct chracter in the substring
            # should be less than or equal to k to be valid, since we can only flip k characters
            # aka the substring length (ind - start + 1) equal or less than maxfreq + k
            # so if not valid
            if not ind - start + 1 - max_freq <= k:
                freq_counter[s[start]] -= 1
                start += 1
            
            # Commented out this part because it's actually unecessary since the window size won't ever shrink
            # max_length = ind - start + 1
        
        return len(s) - start
```        
        
        





