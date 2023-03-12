# 136, Single Number

## Problem Statement
---
> Problem I: Given a non-empty array of integers nums, every element appears twice except for one. Find that single one. You must implement a solution with a linear runtime complexity and use only constant extra space.
---

## Thought Process

This one is simple; you just need to remember that bits exist and the different ways to manipulate them. Comparing two integers with an XOR operation (exclusive or), the result will only have 1 in the areas where only one of the inputs had a 1. e.g (1010110 ^ 0111010 = 1101100). So if you did an XOR operation on two of the same integers, you'd get 0. (1101 ^ 1101 = 0000). Using that same logic, an XOR between 0 and an integer would return the integer (1011 ^ 0000 = 1011). 

Therefore, you can loop through the array and constantly add each number to our answer bit. When we eventually reach its copy, they'll cancel each other out in the bit, until we're just left with 0 ^ the integer that has no duplicate. Ta-da. 

The time complexity is then linear, with constant space complexity.

```
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans ^= num
        return ans
```

# 191, Number of 1 Bits

## Problem Statement
---
> Problem I: Write a function that takes the binary representation of an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).
>
> Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned. In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.
---

## Thought Process

A signed integer is a 32-bit datum that encodes both positive and negative values (aka 'signed'). An unsigned integer is a 32-bit datum that encodes **non-negative** integers (Note that this includes 0). Also, say that the signed integer has a range (-x, x); the unsigned integer would have a range (0, 2x).

For this problem, we can loop through all 32-bits and check each one if they're turned on, but there's an easy way to optimize it a bit further, even if the time complexity ultimately is the same. We can turn off the least significant bit!

There are two ways to turn off the rightmost instance of 1: we can do the & operation with (n - 1), or we can get the least significant bit as its own separate value through (n & -n) and then subtracting that value from n. Basically, we can either just directly turn off the bit, or we can get that bit and subtract it from the orginal num. Each time we do this, we increment the counter by 1, and we stop once the number equals zero.

Note: This does require modifying the input num. If the interviewer doesn't want us to modify the input, we can mke a deep copy of the value (since python's annoying about just taking the value and not the reference) or we can go with the 32-bit loop approach.

O(N) time complexity (technically O(1) since we know it's always 32 bits). O(1) space complexity

```
class Solution:
    def looped_hammingWeight(self, n: int) -> int:
        bit_counter = 0
        mask = 1
        for i in range(32):
            if (mask & n) != 0: 
                bit_counter += 1
            mask <<= 1
        return bit_counter
        
  def optimal_hammingWeight(self, n: int) -> int:
        counter = 0
        while n:
            # Either of the below two operations would work for our purposes
            # n &= n - 1
            n -= (n & -n)
            counter += 1
        return counter
```

# 338, Counting Bits

## Problem Statement
---
> Problem I: Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
---

## Thought Process

Oh god this has so many ways of solving it. You could use the solution we got for the last problem and just run that on every single number in the range(n), but that gives us an O(NlogN) time complexity, and there are at least 3 other O(N) DP solutions. All the solutions still take constant space however.

One thing to note, an integer mod 2 is the same as saying integer & 1. So x % 2 = x & 1. Which makes sense cause binary. Fuck binary. All my homies hate binary. The below picture explains it really well, but you can also optimize it further.
![image](https://user-images.githubusercontent.com/118993796/224515850-164507b7-d87f-4a0a-b94e-ccc5ca1ce9f9.png)

The more optimized solution also feels more intuitive. If you turn off the least significant bit (so the last 1), then you can check that index for how many 1's were detected. Taking that, you just add 1 to get the count for your current index, since you turned off a single 1-bit to get that index value. This works since we're building up the array continuously with all the previous values up to the target integer n.

```
class Solution:
    def dp_leastsignificant_countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans
    
    # Most optimal
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i & (i - 1)] + 1
        return ans
```

# 190, Reverse Bits

## Problem Statement
---
> Problem I: Reverse bits of a given 32 bits unsigned integer.
---

## Thought Process

Anotehr seemingly easy one that I complicated because I was trying to flip the bits instead of reversing the order. Cool lessons from this problem: Using the or operation a|b is equivalent to a + b when it comes to combining the bits. (at least for the rightmost one, afaik). 

The first solution was to create a new variable and to loop through the 32-bit range, updating the rightmost bit of the new variable with the rightmost bit of the original input num. The only difference would be that we shift the new variable to the left after we're done, while the input is shifted to the right. There's also a neat offshoot of this approach that builds from the left to the right instead of right to the left. I'll include both.

I'll be honest, I just don't understand the solutions that don't use a for-loop. I can't wrap my head around them since I also barely understand bits. 

```
class Solution:
    def reverseBits(self, n: int) -> int:
        # Approach 1 from left to right:
        result = 0
        for i in range(31, -1, -1):
            result += (n & 1) << i
            n >>= 1
        return result
        
        # Approach 1 building from right to left
        switch = 0
        for i in range(32):
            switch <<= 1
            switch |= n & 1
            n >>= 1
        return switch

class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        ret, power = 0, 24
        while n:
            ret += self.reverseByte(n & 0xff) << power
            n = n >> 8
            power -= 8
        return ret

    # memoization with decorator
    @functools.lru_cache(maxsize=256)
    def reverseByte(self, byte):
        return (byte * 0x0202020202 & 0x010884422010) % 1023
```

# 268, Missing Number

## Problem Statement
---
> Problem I: Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
---

## Thought Process

This one's pretty similar to the first one that we solved. We can use XOR to cross out duplicates. The key is to recognize that the array is of size n, but the range is n + 1 (from 0 to n). Logically, from a reverse of the pigeonhole principle, that means that exactly one element is missing, since the numbers are distinct, but that information is given to us anyways. But if we XOR all the values in that range, 0 through n, and then loop through the array and XOR all of those elements, the repeated elements will be crossed out and we'll be left with the missing element. I honestly like this one, it's kinda elegant and fun. 

And you can also just be cheeky and use Gauss' formula of n * (n+1) / 2, but that's still linear time. Both approaches are O(N) time and O(1) space.

```
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        result = len(nums)
        for i in range(result):
            result ^= i ^ nums[i]
        return result
        
    def Gauss_missingNumber(self, nums):
        expected_sum = len(nums)*(len(nums)+1)//2
        actual_sum = sum(nums)
        return expected_sum - actual_sum
```

# 371, Sum of Two Integers

## Problem Statement
---
> Problem I: Given two integers a and b, return the sum of the two integers without using the operators + and -.
---

## Thought Process

This problem's incredibly stupid and I'm mad at it. It's also the first Leetcode problem I've seen tih nearly 50% more dislikes than likes. I honestly don't even want to explain it; I'm so fed up with it. If I'm looking back at this one, remember that it has something to do with using XOR and answers without carries. Maybe I need to watch neetcode's video on it. Yeah it's still kinda rough, but it makes a bit more sense. Idk, I feel like this is one of the ones you just memorize. For this one, watch the neetcode video, and then check the leetcode optimal solution below. Then look at the image below and watch the second vid. That'll help a **bit** hahahaha yeah I need psychiatric help. Also google something called half-adder.   https://www.youtube.com/watch?v=gVUrDV4tZfY&t=66s
![image](https://user-images.githubusercontent.com/118993796/224523464-7e1a60eb-762d-46d3-bc93-cc75beb9fef7.png)
https://www.youtube.com/watch?v=4qH4unVtJkE&t=1s

```
class Solution:
    def LeetcodeOptimal_getSum(self, a: int, b: int) -> int:
        # 32 1-bit bitmask
        mask = 0xFFFFFFFF

        while b != 0:
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask
        
        # 31 1-bits, if it's greater there is an overflow
        max_int = 0x7FFFFFFF
        return a if a < max_int else ~(a ^ mask)
        
    def Neetcode_getSum(self, a: int, b: int) -> int:
        def add(a, b):
            if not a or not b:
                return a or b
            return add(a ^ b, (a & b) << 1)

        if a * b < 0:  # assume a < 0, b > 0
            if a > 0:
                return self.getSum(b, a)
            if add(~a, 1) == b:  # -a == b
                return 0
            if add(~a, 1) < b:  # -a < b
                return add(~add(add(~a, 1), add(~b, 1)), 1)  # -add(-a, -b)

        return add(a, b)  # a*b >= 0 or (-a) > b > 0
```

# 7, Reverse Integer

## Problem Statement
---
> Problem I: Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0. Assume the environment does not allow you to store 64-bit integers (signed or unsigned).
---

## Thought Process

The Leetcode premium solution doesn't even explain a bit manipulation solution. Idk why it's under this category, I gotta watch Neetcode's video. Wait actually it is the bit solution? I'm confused, I thought bits were dealing with pure binary, not modding vals by 10. Honestly, this is another one where the video will be more helpful than me explaining it. Both this video plus the previous problem's video should only take about 15min if I watch at 2x speed, use that to recap. O(logN) tie complexity and contant space. It's log time complexity because there are roughly log10 digits in the input integer.

https://www.youtube.com/watch?v=HAgLH58IgJQ&t=4s

```
class Solution:
    def reverse(self, x: int) -> int:
        MAX = 2**31 - 1 # 2147482647
        MIN = -2**31 # -2147482648
        
        result = 0
        while x:
            last_dig = int(math.fmod(x, 10)) # python is butts, -1 % 10 = 9 acording to python, so hence this function
            x = int(x / 10) # we use int() instead of // because x can be negative, and we want to go towards 0
            if (result > MAX // 10 or (result == MAX // 10 and last_dig >= MAX % 10)): return 0
            if (result < MIN // 10 or (result == MIN // 10 and last_dig <= MIN % 10)): return 0
            result = result * 10 + last_dig
        return result

```
