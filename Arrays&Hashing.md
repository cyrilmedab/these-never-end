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
> 
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

Honestly nah, fuck that. If you want to learn this one, google it yourself, ya dumbass. Here's a comment that helped me understand tho ![image](https://user-images.githubusercontent.com/118993796/224221476-0485eb9f-ac96-4dc7-8b98-cb74d6f793f6.png)

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

# 347, Top K Frequent Elements

## Problem Statement
---
> Problem I: Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.
---

## Thought Process

There's a heap solution to this with O(Nlogk) time complexity and the same linear space complexity as the quick select method, but I don't really want to bother with that when honestly, quick select should be my first though when looking at these kind of problems. 

I need to drill quick select and lomuto's partition into my head until I start having sexually confusiing dreams about them. It's not that difficult, but for some reason, it keeps slipping out of my mind every time. Quick select follows the same process as quicksort, a sorting algorithm with O(NlogN) time complexity. However, because we're only going to care about half of the array (the kth section), we don't need to sort the entire array and can reduce it down to linear time **in the average case**. In the worst case, where the partion keeps selecting the largest element and having to sort through the entire array, quick select runs in O(N^2) time. However, this is negligible for the most part.

Lomuto's partition is a really simple parition algorithm that will select a random pivot. I instead just choose the rightmost point as the pivot index, but as shown in Leetcode's official solution, you can randomize this selection very easily. The only extra step after randomizing the index would be swapping the random index's position with the rightmost position.

We want to find the kth most frequent elements in the array. We can clarify with the interviewer that k must be less than the number of unique elements in the array. We can also ask how we should be handling ties; would we need to only include one of them, can we include all the elements that tie, and if we can only include of the tied elements, is there a criteria for selecting which one we include.

The step after that would be to figure out the frequencies. We can iterate through the array and add each new value we see to a dictionary of ints. Each time we see a value that is already in the dictionary, we can increment the value paired with that key by one. So after the iteration, we then have a dictionary containing every unique element and the amount of times each appeared in the input array. The last step for this part is to take all the keys of the dictionary and store them in a new list; this gives us a list with only unique elements, like a set. We don't want to use a set, however, because then we can't iterate through it.

For quickselect, we know that the index that the kth most frequent element will be at (when the array is sorted) will be the length of the new list we just created - k. We can sotre that really quickly, and then initialize two pointers at the first and last index of the list.

You can create separate helper functions for the partions and quickselect, like Leetcode did, and you can also do it recursively, but I found my solution a bit easier to follow and simpler. The folowing is a quick reminder of quickselect. Google it if you need more details, but hopefully at this point it's either engrained in my memory or I can glance at the code and remember. 

Set another pointer to the leftmost index and store the frequency value of the right pointer's setlist value. Run quickselect and swap values as needed. Once you've iterated through the setlist, you should have the pivot values (the value of the right pointer that we assigned at the start) moved towards a central location. We can then check the three conditionals that are critical to quickselect.

If the pivot index now equals the kth index that we're looking for, we can return all the values from the pivot index to the end of the setlist array. If the kth index is greater (meaning that there should be less elements that we'll eventually be returning), we'll need to again sort through the right half of the array, and we can move our left pointer to the pivot index + 1. And the exact opposite if the kth index is less than the pivot index. We need to keep sorting the left half, and should move the right pointer to the pivot index - 1.

```
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # This builds a counter of how many times a given element appears in the input array
        # and can be used as our "value assessor" for the quick select
        frequencies = collections.defaultdict(int)
        for num in nums:
            frequencies[num] += 1
        nums_setlist = list(frequencies.keys())

        k_index = len(nums_setlist) - k
        left, right = 0, len(nums_setlist) - 1
        
        
        while left < right:
            point_ind = left
            point_freq = frequencies[nums_setlist[right]]

            for i in range(left, right):
                if frequencies[nums_setlist[i]] < point_freq:
                    nums_setlist[i], nums_setlist[point_ind] = nums_setlist[point_ind], nums_setlist[i]
                    point_ind += 1

            nums_setlist[point_ind], nums_setlist[right] = nums_setlist[right], nums_setlist[point_ind]
            if point_ind == k_index:
                return nums_setlist[k_index: ]
            elif point_ind > k_index:
                right = point_ind - 1
            elif point_ind < k_index:
                left = point_ind + 1

        return nums_setlist[k_index: ]

from collections import Counter
class Leetcode_Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        unique = list(count.keys())
        
        def partition(left, right, pivot_index) -> int:
            pivot_frequency = count[unique[pivot_index]]
            # 1. move pivot to end
            unique[pivot_index], unique[right] = unique[right], unique[pivot_index]  
            
            # 2. move all less frequent elements to the left
            store_index = left
            for i in range(left, right):
                if count[unique[i]] < pivot_frequency:
                    unique[store_index], unique[i] = unique[i], unique[store_index]
                    store_index += 1

            # 3. move pivot to its final place
            unique[right], unique[store_index] = unique[store_index], unique[right]  
            
            return store_index
        
        def quickselect(left, right, k_smallest) -> None:
            """
            Sort a list within left..right till kth less frequent element
            takes its place. 
            """
            # base case: the list contains only one element
            if left == right: 
                return
            
            # select a random pivot_index
            pivot_index = random.randint(left, right)     
                            
            # find the pivot position in a sorted list   
            pivot_index = partition(left, right, pivot_index)
            
            # if the pivot is in its final sorted position
            if k_smallest == pivot_index:
                 return 
            # go left
            elif k_smallest < pivot_index:
                quickselect(left, pivot_index - 1, k_smallest)
            # go right
            else:
                quickselect(pivot_index + 1, right, k_smallest)
         
        n = len(unique) 
        # kth top frequent element is (n - k)th less frequent.
        # Do a partial sort: from less frequent to the most frequent, till
        # (n - k)th less frequent element takes its place (n - k) in a sorted array. 
        # All element on the left are less frequent.
        # All the elements on the right are more frequent.  
        quickselect(0, n - 1, n - k)
        # Return top k frequent elements
        return unique[n - k:]
```

# 36, Valid Sudoku

## Problem Statement:
---
> Problem I: Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules: Each row must contain the digits 1-9 without repetition. Each column must contain the digits 1-9 without repetition. Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
> 
> Note: A Sudoku board (partially filled) could be valid but is not necessarily solvable. Only the filled cells need to be validated according to the mentioned rules.
---

## Thought Process

This one seems really intimidating, and maybe that's the goal, but your first intuition might be right in this case: we're using 3 separate dictionaries of sets, for the rows, columns, and squares. 

After accepting that you have to use these large dictionaries to store everything, the most difficult part is how to properly index the square grids, but that's easily solved by **floor dividing** the row and column we're looking at by 3. Floor diving means that we're getting the lowest integer and not any decimals. So let's say we look at the first three rows in the first column: (0,0), (1,0), and (2, 0). Floor dividing them by three gives us (0,0), (0,0), and (0,0). So they'd all be in the same grid, as we can see. If we looked at the 4th row (which is index 3) the 3 would divide to get 1, meaning those would be stored in a different square reference in our dictionary. We want to use both the row and column value to get our keys for the square dictionary, which means we should store them as a tuple. 

After figuring all this out, we just need to iterate through every grid in the array using 2 for-loops. If we hit a value, we can check if they're in any of the sets. If they aren't we add them to all three of the sets at their respective values. If they are in any one of the sets, we instantly return False, since that can't occur for a valid sudoku board.

Finally, we can return true if we've gone through every grid and haven't found a violation that made us return false.

Note: This gives us an O(N^2) time complexity and requires an O(N^2) space complexity as well, although you can easily argue that both are constant since N will always be 9. Technically, you can reduce the space complexity to O(N) by passing over the board 3 times instead of trying to do everything in a single pass. So this would in actuality greatly increase the performance time of the algorithm but it technically keeps the same time complexity. If you went with this constant space solution, you would reset the set each time you check a row. Then move on to the columns, resetting after each finished column. It's essentially like brute forcing the solution. 

There is an O(N) space complexity using bitmasking that I need to learn. Neetcode didn't even mention this approach, so I didn't notice it until looking through Leetcode Premium's solution. I haven't attempted it, so let's go over it here.

Actually holy fuck, I love the bitmasking solution, it's neat. Full disclosure, I honestly don't know how to work with bits yet, I'm still working my way through all the data structures, so this is probably pretty normal and expected. BUT, with the bit masking, you can solve this in a single pass over the grid with an O(N^2) time complexity and an O(N) space complexity. AND, if you do it in a triple pass, you could technically solve it with the same time complexity but a constant O(1) space complexity. Shit's awesome. I don't want to type it all out tho, so here's a pic: ![image](https://user-images.githubusercontent.com/118993796/224460089-3b66877b-4ae3-4899-833f-a95ecccd78eb.png)

I'm also going to include the hard Leetcode problem to solve a sudoku board, just cause I liked it and found this one interseting. It's a backtracking problemt that's very straightforward. 

```
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = collections.defaultdict(set)
        cols = collections.defaultdict(set)
        squares = collections.defaultdict(set)

        for row in range(len(board)):
            for col in range(len(board[0])):
                curr_val = board[row][col]
                if curr_val == ".":
                    continue
                if curr_val in rows[row] or curr_val in cols[col] or curr_val in squares[(row//3, col//3)]:
                    return False
                rows[row].add(curr_val)
                cols[col].add(curr_val)
                squares[(row//3, col//3)].add(curr_val)
        return True
    
    # Bitmasking solution
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        size = len(board)

        rows = [0] * size
        cols = [0] * size
        squares = [0] * size

        for row in range(size):
            for col in range(size):
                if board[row][col] == ".":
                    continue
                
                val = int(board[row][col]) - 1

                box_index = (row // 3) * 3 + (col // 3)
                if (rows[row] & (1 << val)) or (cols[col] & (1 << val)) or (squares[box_index] & (1 << val)):
                    return False

                rows[row] |= 1 << val
                cols[col] |= 1 << val
                squares[box_index] |= 1 << val
                
        return True
        
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def can_place(num, row, col):
            return not (num in rows[row] or num in cols[col] or \
                    num in squares[box_index(row, col)])
        
        def place_num(num, row, col):
            rows[row].add(num)
            cols[col].add(num)
            squares[box_index(row, col)].add(num)
            board[row][col] = str(num)
        
        def remove_num(num, row, col):
            rows[row].remove(num)
            cols[col].remove(num)
            squares[box_index(row, col)].remove(num)
            board[row][col] = '.'
        
        def place_next(row, col):
            if row == N - 1 and col == N - 1:
                nonlocal sudoku_solved
                sudoku_solved = True
            else:
                if col == N - 1:
                    backtrack(row + 1, 0)
                else:
                    backtrack(row, col + 1)
        
        def backtrack(row = 0, col = 0):
            if board[row][col] == '.':
                for num in range(1, 10):
                    if can_place(num, row, col):
                        place_num(num, row, col)
                        place_next(row, col)
                        if not sudoku_solved:
                            remove_num(num, row, col)
            else:
                place_next(row, col)
        
        n = 3
        N = n**2
        box_index = lambda row, col: (row // n) * n + (col // n)

        rows = [set() for i in range(N)]
        cols = [set() for i in range(N)]
        squares = [set() for i in range(N)]
        for row in range(N):
            for col in range(N):
                if board[row][col] == '.': 
                    continue
                num = int(board[row][col])
                place_num(num, row, col)
        
        sudoku_solved = False
        backtrack() 
```

# 128, Longest Consecutive Sequence

## Problem Statement
---
> Problem I: Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence. You must write an algorithm that runs in O(n) time.
---

## Thought Process 

The last of the Arrays and Hashing problems (from Neeetcode's 150). I feel like this is definitely more tricky than it seems, but like most of the previous problems, the answer lies with (gasp) sets. Also, ik I'm incredibly stupid, but I legitimately just processed that the reason so many of these solutions involved sets is because this is a section about Arrays and Hashing. Hashing. As in, hash sets. And I just put this together. Do I not have basic reading comprehension like dude.

Anyways, you can drop all of the elements in the given input array into a set. Then, we can do a for-loop for every element in the set. At each element, we're going to set a counter value to 1 (representing the element that we're on being a sequence on its own). Then we can enter a while lopp and check to see if the number after our current num (num + 1) is in the set. If it is, then we increment our counter and check if num + 2, etc. Once we break out of the while loop, we can check to see if our counter is greeater than our stored value for the max length.

At this point, we have an O(N^2) solution. This is because our algorithm checks every instance in a chain multiple times. For example, take [1, 2, 3, 4]. Our algorithm, starting at 1, will detect a max length of 4. But then it continues and starting at 2, detects a max length of 3. But we already know that 2, 3, and 4 won't beat our max length because they were already part of the sequence that did. How do we avoid checking them?

We can reduce our time complexity down to O(N) by adding an if-statement that only checks for the sequence length if the number before the current num (aka num - 1) isn't in the set. That means we'll only bother with each element in the set once.

```
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        uniques = set(nums)

        max_length = 0
        for num in uniques:
            if num - 1 in uniques: continue
            
            curr_length = 0
            while num in uniques:
                curr_length += 1
                num += 1
            max_length = max(max_length, curr_length)
        return max_length
```
