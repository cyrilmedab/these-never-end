# 217, 219 Contains Duplicate I & II

## Problem Statement
---
> Problem I: Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
>
> Problem II: Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
---

## Thought Process

217 is the one that's included in the Blind 75, not 219. However, I feel that 217 is so aggressively easy; the solution feels like it's literally to remember that sets exist. My suspicion is that's actually part of the challenge when this one is brought up in coding interviews, cause Leetcode suggests that it's used surprisingly often.

The problem is so generic that it could be worded in different contexts, given different meanings, have a lot of filler information--all to obscure the fact that the solution is incredibly simple. So what the interviewer would be testing is your ability to parse information and isolate the important bits. 

Anyways, cause it's too simple, I included the next stage of the problem, which while still being an easy problem, has a slightly more fun twist.

Reiterating, the solution to the first problem is to use a set. Sets give us constant time lookup and will only store a distinct value once; our set can't contain the same value twice. Working through the given array then, we can add each value in the array to the set. Each time we add a value though, we also want to check if the value is already in the set. If it's in the set, we satisfy our goal of detecting if a number appears **at least twice** and can return true. We don't care about what number it is or which index we found it in or anything else.

The next problem asks us a similar question, except this time, we can see that we have to keep track of the indices as well. Sets aren't up to the task, so we can use the heavier hitter: dictionaries. We can use the integers in the given input array as our dictionary keys, and store the index where we found them in their value. 

Once again, we can loop through the input array, storing the index of each integer we see in our dictionary. At this point, we can use our dictionary to look up a specific value and the index where we saw it. For each number, we also want to check if it satisfies our condition! If the number is already in the dictionary (as a part of the keys), then we can subtract the stored index of our current value from the current index of our current value. If that's <= k, then we can return true.

An important thing to note is that we don't even have to compute the absolute value of the difference. Since we're progressing from left to right, we know that our current index will always be greater than our stored index. Therefore, the absolute value calculation is unecessary if we put current index - stored index.

Another question is if we have more than just two copies of a value in the input array. Do we need to store all the indices and check our wincon against all of them each time we find a new repeated value? No. We will only care about the most recently seen index of the current number. Any prior appearances would end up having a greater difference in the index values, so are unnecessary to check against.

All that said, I'm an idiot and Leetcode Premium shows that you can solve the second problem even more efficiently with a smaller space complexity using just a set. You do this by combining a set with a sliding window approach. The set will only contain the k most recent elements; we can add each elemnt we see to the set and remove the oldest value (aka the value of num[curr_index - k]) from the set. This gives us the same linear time complexity of the dictionary approach but a better space complexity. Instead of O(N), we get O(min(N, k)). This is because our set will require k space but never more than N space.

 Note that while the time complexity is technically the same, leetcode submission show that the dictionary approach regularly runs faster. Not important for interviews, just cool.

```
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
        
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        seen_indices = {}

        for i in range(len(nums)):
            if nums[i] in seen_indices:
                if i - seen_indices[nums[i]] <= k:
                    return True
            seen_indices[nums[i]] = i
        return False
        
   # The set approach
   def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        seen_indices = set()

        for i in range(len(nums)):
            if nums[i] in seen_indices:
                return True
            seen_indices.add(nums[i])
            if len(seen_indices) > k:
                seen_indices.remove(nums[i - k])
        return False
```

# 242, 49 Valid Anagram & Group Anagrams

## Problem Statement
---
> Problem I: Given two strings s and t, return true if t is an anagram of s, and false otherwise. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
> Problem II: Given two strings s and t, return true if t is an anagram of s, and false otherwise. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
---

## Thought Process

This one's kinda fun imo. My initial intuition was to use a dictionary, which I still stand by, but the official solutions use an array to represent the characters. The dictionary solution gives a little more flexibility and can potentially occupy less space, but the array might be faster? Leetcode's submission is being funky and giving my solutions insanely low runtimes rn, despite them being faster earlier. Both have the same time complexity, and you can argue that the dictionary solution is still constant space, since it will never have more than 26 keys, for the 26 characters of the alphabet. That actually brings up a really good question taht you should definitely ask in interviews.

The most important step for both of these problems is to discuss constraints. Are the characters all going to be lower case? If not, does case matter? Can we expect unicode characters or ASCII? Can we confirm that we'll only see a certain range of ASCII characters (the 26 lowercase alphabetical characters in this problem's case). The leetcode suggestd follow-up for the first problem is how we'de adjust our problem if we were to use Unicode characters instead. This is where you can't feasibly use the suggested array strategy and you have to use the dictionary strategy I provide below. This is because there are too many characters in Unicode, and it's impractical to create a fixed array of size 149,186 for unicode. Anyways, on to the solution.

The simplest solution for the first problem (and definitely the quickest to code) is to simply sort both input strings and check if they're equal. It's worth noting this solution technically has constant space, but this can be language dependent and is affected by the sorting algorithm that is used. The sort will also run in best-case O(NlogN) time, and we can definitely do better than that.

We're only looking at the lowercase alphabet characters for both of these problems, so from 'a' to 'z'. We can create a fixed array of size 26, and using the ASCII values of each character, we can make sure that each character references a different index in the 26-sized array (The code makes this clearer than I can explain). So looping through one of the input strings, we can increment the values in this array by 1, allowing us to count how many times a character appears in the string. Now if we look at the second string and loop through that, we can actually decrement the values in the array we created whenver we detected a character. This is because anagrams must have the same character count. Therefore, if we ever decrement a chracter's count and end up with a value lower than 0, we instantly know that the two strings can't be anagrams of each other.

The dictionary solution operates on the same principle, just substituting the array for a dictionary. This would be more optimal in some situations. For example, a string with a length of 100 but only contains the character 'a' would still require an array of length 26. The dictionary approach would only have a size of 1, with 1 key-value pair.

The second problem statement is a bit more tricky, and while we can still use a dictionary here, I'm only going to use an array to keepy it simple. Remember, anagrams must have the exact same chracter count. Therefore, using the approach we described above, anagrams would also for the same array when we count their characters. These arrays can be used to group words that have the same chracter count then. Dictionaries are the perfect data structure for this. We can append a given word to the dictionary by using it's character count array as the key. Then we just have to return all the values in the dictionary.

Note: we cast the keys for this dictionary as a tuple, since we aren't allowed to use lists as keys for python dictionaries.

```
class Solution:
    def sorting_isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)
        
    def myArraySolution_isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        char_array = [0] * 26

        for char in s:
            char_array[ord(char) - ord('a')] += 1
        
        for char in t:
            char_index = ord(char) - ord('a')
            char_array[char_index] -= 1
            if char_array[char_index] < 0:
                return False

        return True
        
    # I think Neetcode's solution here wastes unnecessary space by creating a second dictionary for the second string
    def Neetcode_isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        countS, countT = {}, {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        return countS == countT
        
    def dictionaries_isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        s_chars = {}
        for char in s:
            if char in s_chars:
                s_chars[char] += 1
            else:
                s_chars[char] = 1
        
        for char in t:
            if char not in s_chars:
                return False
            s_chars[char] -= 1
            if s_chars[char] < 0:
                return False
        return True
        
    # PROBLEM 2
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        storage = collections.defaultdict(list) # This just allows us to initialize our dictionary with empty lists
        
        for word in strs:
            code = [0] * 26
            for char in word:
                code[ord(char) - ord('a')] += 1
            storage[tuple(code)].append(word)
            # Below is how we'd add to the dictionary if we used a empty dictionary and didn't use collections.defaultdict(list)
            #if not tuple(code) in storage:
            #    storage[tuple(code)] = [word]
            #else:
            #    storage[tuple(code)].append(word)

        return list(storage.values())
```

# 1, Two Sum

## Problem Statement
---
> Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.
---

## Thought Process 
Leetcode numero uno. I swear, I'll cry tears of joy if I get asked this question in an interview. Super simple concept, and once you see it, it's hard to forget.

If you're looking for two numbers (let's call them x and y) that add up to a target value, you then have the equation x + y = target. From that, you can derive that y = target - x. Every value of x has a paired complement that adds up to y, and you can get that complement by subtracting x from the target. So if we know the complement that we want to find, we just have to store the indices.

We can loop through each value of the array and compute what the complement value should be for our current integer. We can then store the current index in the dictionary, using the complement as the key. Before we store that data however, we should check if the complement is already in the array. If it is, we can return the stored index as well as the current index we're on in an array. And that's it. You solve this in O(N) time and O(N) space. 

A neat thing to ask (possibly?) is if the inteviewer would like us to prioritize a better space or time complexity in this case. For example, the brute force for this approach runs in O(N^2) time, but constant space. Idk, up to y'all, you random figments of ym imagination that I'm typing out shitty conversations too. 

God I'm lonely.

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        complements = {}
        for i in range(len(nums)):
            if nums[i] in complements:
                return [complements[nums[i]], i]
            complements[target - nums[i]] = i
```

# 238, Product of Array Except Self

## Problem Statement
---
> Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]. The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. You must write an algorithm that runs in O(n) time and without using the division operation.
---

## Thought Process

A difficult perspective to achieve, but actually another super easy problem once you see it. You want the products of everything but the element of your index. What would be the value of the last element in your index? It would be the product of everything that came before it, which we could easily calculate in one pass. What about for the first index? It would the the product of everything after it **or from a different perspective aka working backwards on the array** it would be the product of everything before it. Computing these two values would give us intermediary products at each index of everything before/after the current index. **Before/After**. So not the current index! Therefore we already have our solution. Contructing these two arrays and multiplying the values at each indices gives us our solution.

The above solution requires O(N) space, but we can do this in constant space. If the above explanation didn't make it clear, here's another way of looking at it. You lag behind on the product calculation. The value you;re using to store the total product information is always copied into the solution array before you multiply the current input array value into the total. If you do this from left to right, reset the storage, and then do it right to left, you have a solution in constant space, since the result array does not count towards space complexity.

```
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        product_array = []
        prev = 1
        for num in nums:
            product_array.append(prev)
            prev *= num
        
        prev = nums[-1]
        for ind in range(len(nums) - 2, -1, -1):
            product_array[ind] *= prev
            prev *= nums[ind]
        return product_array
```

# 271, Encode and Decode Strings

## Problem Statement
--- 
> Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.
> 
> Machine 1 (sender) has the function:
```
string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}
```
> Machine 2 (receiver) has the function:
```
vector<string> decode(string s) {
  //... your code
  return strs;
}
```
> strs2 in Machine 2 should be the same as strs in Machine 1. Implement the encode and decode methods. You are not allowed to solve the problem using any serialize methods (such as eval).
---

## Thought Process

Skipped to this one because I'm avoidng the K Frequent problem until the morning when I'm less psychotic and don't feel like punching Hoare and Lomuto in the face. I'll show them a stupid partition. 

Encode and Decode is another one of those problems where you really need to clarify with the interviewer if you're working with the 256 ASCII characters or the whole buttload of Unicode characters. Mostly to show you're paying attention, because we're going to create a generalized encoding algorithm that's super simple and can be paired with any characters in any arrangement. 

For encoding, we want to return a single string. So let's just concatenate every string in the given list together. Except that would be really hard to decode; we wouldn't know where one word ends and another begins. Okay fine, lets add a random character or a unique identifier of sorts before each new word we add to the string. That way, our future decoding algorithm will at least be able to tell that a word exists. 

How should we select what the identifier should be? It can't be a single character, since that character could be in the string. Similarly an random sequence could also potentially be in the strings we're trying to encode, however unlikely. The really fun way around this is to store information about the length of the word you're encoding into the partitions we're adding. If we know the length of the word when we're decoding, then it doesn't matter if the identifier string is also contained in the word, because we'll be skipping ahead to the end of the word before we start checking for the identifier again. And that's how we generalize the algorithm to work on any possible set of characters.

Decoding it takes more code but is much more logically simple, given how we designed our encoding algorithm. Starting at the first index of the encoded string, there should be a number (stored as a string) that tells us the length of the following word. The number could be 1, 2, or even 3 digits, so we can loop forwards until we hit the identifier character, storing the string as we go. Once we hit the identifier, the previous stored string can be converted into an int that tells us the word length. We use that information to slice that word out of the string and into our result array, and then move to the end of the word before we start the whole process again. Repeat till you've gone through the enitre encoded string.

This is the easiest approach to come up with and remember, but apparently Google likes when you use chunked transfer encoding. And because bits is just all around pretty confusing for me, I'm going to cover this method here. Cause if it's confusing, it'd probably be hella impressive to pull of in an interview, and would def elevate you're interviewers impression of you if you're confident withh bit manipulation. 

This solution follows the same principle as we did for our initial solution. However, this time we're converting the length of the string/word/chunk into a 4 bytes string. That's all, but the code looks hella confusing. 

Honestly nah, fuck that. If you want to learn this one, google it yourself. Here's a comment that helped me understand tho ![image](https://user-images.githubusercontent.com/118993796/224221476-0485eb9f-ac96-4dc7-8b98-cb74d6f793f6.png)

```
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        result = ""
        for word in strs:
            result += str(len(word)) + "#" + word
        return result
        

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        result = []
        ind = 0
        while ind < len(s):
            start = ind
            while s[ind] != "#":
                ind += 1
            next_len = int(s[start:ind])

            ind += 1
            result.append(s[ind: ind + next_len])
            ind += next_len
        return result
        
# The bytes approach
class BytesCodec:
    def len_to_str(self, x):
        """
        Encodes length of string to bytes string
        """
        x = len(x)
        bytes = [chr(x >> (i * 8) & 0xff) for i in range(4)]
        bytes.reverse()
        bytes_str = ''.join(bytes)
        return bytes_str
    
    def encode(self, strs):
        """Encodes a list of strings to a single string.
        :type strs: List[str]
        :rtype: str
        """
        # encode here is a workaround to fix BE CodecDriver error
        return ''.join(self.len_to_str(x) + x.encode('utf-8') for x in strs)
        
    def str_to_int(self, bytes_str):
        """
        Decodes bytes string to integer.
        """
        result = 0
        for ch in bytes_str:
            result = result * 256 + ord(ch)
        return result
        
    def decode(self, s):
        """Decodes a single string to a list of strings.
        :type s: str
        :rtype: List[str]
        """
        i, n = 0, len(s)
        output = []
        while i < n:
            length = self.str_to_int(s[i: i + 4])
            i += 4
            output.append(s[i: i + length])
            i += length
        return output
```

# Top K Frequent Elements
