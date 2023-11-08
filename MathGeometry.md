# 48, Rotate Image

## Problem Statement
---
> You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
> You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
---

## Thought Process

LeetCode legit recommends memorizing this problem and getting super comfortable with it, so this should get ingrained in your mind and become reflexive. There are two solutions for this, with one being more efficient (half as many read and writes) yet more messy and difficult to follow (so considered worse code). It's important to mention both and discuss the tradeoffs of these.

Super important for the second method based on matrices' mathematical properties: if you're rotating **clockwise**, you want to rotate along the main diagonal (topleft to bottom right)/ If you're rotating 90deg **counter-clockwise**, you just have to transpose across the other diagonal and then it's the same.

Okay, first up is moving in a diagonal pattern through the matrix and swapping the values in groups of 4. This should be relatively intuitive, but the execution is tricky and can easily introduce off-by-1 index errors. 

The pattern relies on the size of the array, with the outer-loop dictated by a ceiling division of half the size, and the ineer loop by a floor division of half the size.  Instead of doing all the outside cells and then moving inwards to a smaller square, this pattern results in cleaner and easier to follow code.

The trick is that our first temp is based on the lower left corner of the matrix, with the outerloop dictating the column and the inner loop moving up in rows. Know this initial placement, deriving the specific pattern for the other three can be done on the spot. 

The second method just involves transposing and reflecting. That's it, nothing much to say there. If you can't remember how to do this, you need to brush up on it, no shortcuts. 

The time complexity for both methods is O(N), though technically the transpose and reflect method is slower since it has a higher constant of 2 instead of the manual's 1. Both solutions require constant O(1) space.

```
class Solution:
    def rotate_manual(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                temp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = temp

    def rotate_mathematical(self, matrix: List[List[int]]) -> None:
        n = len(matrix)

        def transpose() -> None:
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        def reflect() -> None:
            for i in range(n):
                for j in range(n // 2):
                    matrix[i][j], matrix[i][-j - 1] = matrix[i][-j - 1], matrix[i][j]
        
        transpose()
        reflect()
```

# 54, Spiral Matrix

## Problem Statement
---
> Given an m x n matrix, return all elements of the matrix in spiral order.
---

## Thought Process

This one's conceptually simple. One method relies on boundaries for the top, left, bottom, and right. Then it's simply moving along those indices at the boundary and appending the values to a reuslt array. When we finish along the boundaries, we increment or decrement the boundaries by 1 and redo the loop. Slightly messy code with a lot of for-loops, but completely acceptable with no significant performance tradeoffs.

The other method relies on a direction array consiting of the unit parameters when we're moving in a given direction in the matrix. If we have a variable to keep track of when we change direction, we can exit and return our result once we change direction twice back-to-back, since that indicates we no longer have anywhere to go.

This method relies on the usual matrix traversal setup, with a function to determine if a position is within the bounds of the matrix and a visited set. Be careful for the directions array; **the ordering of the directions' array matters**. YOu want it so that you first move right along the columns, then down the rows, then left back across the columns, and finally up the rows.

The time complexities for both is a standard O(MN) where M is the number of rows and N is the number of columns, since we have to visit each element once. The space complexity for the first is O(1), but O(N) space is needed for the second's visited set unless we can modify the input matrix.

```
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows, columns = len(matrix), len(matrix[0])
        spiral = []

        # Boundary Method
        top = left = 0
        right, bottom = columns - 1, rows - 1

        while len(spiral) < rows * columns:
            for col in range(left, right + 1):
                spiral.append(matrix[top][col])
            for row in range(top + 1, bottom + 1):
                spiral.append(matrix[row][right])
            if top != bottom:
                for col in range(right - 1, left - 1, -1):
                    spiral.append(matrix[bottom][col])
            if left != right:
                for row in range(bottom - 1, top, -1):
                    spiral.append(matrix[row][left])
            left += 1; top += 1
            right -= 1; bottom -= 1
        
        # Visited Set Method
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        in_bounds = lambda x, y: 0 <= x < rows and 0 <= y < columns

        row, col = 0, 0
        dir_ind = change_count = 0
        spiral.append(matrix[row][col])
        visited = {(row, col)}

        while change_count < 2:
            while True:
                next_r, next_c = row + directions[dir_ind][0], col + directions[dir_ind][1]
                next_pos = (next_r, next_c)

                if not in_bounds(next_r, next_c) or next_pos in visited:
                    break
                
                change_count = 0
                row, col = next_r, next_c
                spiral.append(matrix[row][col]); visited.add(next_pos)
            change_count += 1; dir_ind = (dir_ind + 1) % 4

        return spiral
```

# 73, Set Matrix Zeroes

## Problem Statement
---
> Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
> You must do it in place.
---

## Thought Process

There's an incredibly simple solution using sets to store the indices of zero'd columns and rows, but there's actually an even better O(1) space solution that keeps the same time complexity O(MN).

We can use the first row and first column as "storage". As we iterate through the array, if we find a 0, we can set the corresponding first row and first column value to zero. Because this would cause errors as we're checking those first areas, however, we want to select one of them to be indicated by a bool (in our code below, we use the first column). Now as we're iterating in the first step, we can check the first value of each row and set the boolean to true if it's 0; this would mean that the entire first column will later be changed to zero.

After the first loop, we can loop through again, but we avoid our indicators (the first row and first column) because we don't want to overwrite them. Once we finish, we can check them individually, with the first row being determined by the (0,0) value and the first column by the bool.

The time complexity is O(MN) where M is the number of rows and N is the number of columns, since we need to read and potentially write each cell. The space complexity is a constant O(1), with only the boolean value really needed for storage!

```
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows, columns = len(matrix), len(matrix[0])
        zeroed_col = False

        for row in range(rows):
            if not matrix[row][0]: zeroed_col = True
            for col in range(1, columns):
                if matrix[row][col]: continue
                matrix[row][0], matrix[0][col] = 0, 0
        
        for row in range(1, rows):
            for col in range(1, columns):
                if matrix[row][0] and matrix[0][col]: continue
                matrix[row][col] = 0
        
        if not matrix[0][0]: 
            for col in range(1, columns):
                matrix[0][col] = 0
        if zeroed_col:
            for row in range(rows):
                matrix[row][0] = 0
```

# 202, Happy Number

## Problem Statement
---
> Write an algorithm to determine if a number n is happy. A happy number is a number defined by the following process:
> Starting with any positive integer, replace the number by the sum of the squares of its digits.
> Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
> Those numbers for which this process ends in 1 are happy.
> Return true if n is a happy number, and false if not.
---

## Thought Process

The first step is to define a helper function that can get the next number in the sequence, so grabbing each digit of a number, squaring it, and summing them together to get our new number. The only input needed for this is the old number, but I've added another input to dictate how many times we want to repeat it (for the efficient solving method that we'll discuss below)

The first solution (which I won't show the code for since it's very similar and easily derivable) relies on a set to store each number that we see. We can run a while loop and return once we either hit a number we've seen before (indicating that we're in a cycle) or we hit 1 (indicating a happy number). This unfortunately requires O(logN) space for the set tho, which we'll explain at the end cause it's complicated.

The optimal solution uses Floyd's Cycle detection algorithm. Even though this is usually applied to linked lists, we're actually operating on a makeshift linked list, where the pointer to the next "node" is our get_next function. Once we realize this, it's simply about having a fast and slow runner and letting the algorithm run until we hit 1 or the pointers are equal. 

There's an even further optimization that actually takes advantage of the fact that there's only 1 cycle that you can encounter, but it's pretty advanced and should not be expected in any interviews, even if it's kinda interesting and is a good read.

Even tho this seems pretty simple, there's a lot that we have to explain to interviewers. One important point is that we have to justify not needing the handle the case where the number keeps going higher and higher towards infinity! Leetcode has a great table visualing this, but the concept is simple. If you have a one digit number, the largest it can be is 9, so the largest next number is 81. For two digits, that's 99 into 162. 3 digits is 999 into 243, 4 is 9999 into 324, and 13 is 9999999999999 into 1053. We can clearly see that there will be a logarithmic decrease from the number of digits, and that once we hit 3 digits, we'll never go above 3. 

I'm just copying the time complexity analysis from Leetcode for this because it's actually a good explanation. The space complexity for our cycle method is O(1). 

Time complexity : O(243⋅3+log⁡n+log⁡log⁡n+log⁡log⁡log⁡n)... = O(log⁡n). Finding the next value for a given number has a cost of O(log⁡n) because we are processing each digit in the number, and the number of digits in a number is given by log⁡n. To work out the total time complexity, we'll need to think carefully about how many numbers are in the chain, and how big they are.

We determined above that once a number is below 243, it is impossible for it to go back up above 243. Therefore, based on our very shallow analysis we know for sure that once a number is below 243, it is impossible for it to take more than another 243 steps to terminate. Each of these numbers has at most 3 digits. With a little more analysis, we could replace the 243 with the length of the longest number chain below 243, however because the constant doesn't matter anyway, we won't worry about it.

For an n above 243, we need to consider the cost of each number in the chain that is above 243. With a little math, we can show that in the worst case, these costs will be O(log⁡n)+O(log⁡log⁡n)+O(log⁡log⁡log⁡n)... Luckily for us, the O(log⁡n) is the dominating part, and the others are all tiny in comparison (collectively, they add up to less than log⁡n), so we can ignore them.

```
class Solution:
    def isHappy(self, n: int) -> bool:
        def get_next(num: int, repeat = 1) -> int:
            for _ in range(repeat):
                new = 0
                while num:
                    num, digit = divmod(num, 10)
                    new += digit ** 2
                num = new
            return num
        
        slow, fast = n, get_next(n)
        while fast != 1 and slow != fast:
            slow, fast = get_next(slow), get_next(fast, 2)
        return fast == 1
```

# 66, Plus One

## Problem Statement
---
> You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.
> Increment the large integer by one and return the resulting array of digits.
---

## Thought Process

Relatively simple problem that's more about not making a silly mistake of forgetting an edge case. Assume you cannot convert to an int, even if you're using Python and don't really have to worry about integer overflow. 

If the last digit isn't a 9, this is a trivial problem where we just increment the last index by 0. The major case we want to handle is a sequence of 9s and when every single digit is a 9. We can do this easily with a for-loop and a variable to keep track of the carry. Since the carry will either be 1 or 0 (the max number we can get from 2 digits added together is 9+9=18), we can exit once the carry is 0. 

**Clarify if we can modify the input, or if we want to return a new array.** If we want to return the input array and modify it in-place, we have to insert the carry at the start when each digit is 9. If we're creating a new array, this isn't a concern anymore.

The time complexity is O(N) where N is the number of digits, and the space complexity is O(1) if we are modifying the input. If we're not modifying the input, you cna argue that we don't include the result in our complexity analysis so it's still O(1), but you obviously need O(N) space for the new result.

```
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        carry = 1
        for ind in range(len(digits) - 1, -1, -1):
            carry, digits[ind] = divmod(digits[ind] + carry, 10)
            if not carry: break
        else: digits.insert(0, 1)

        return digits
```

# 50, Pow(x, n)

## Problem Statement
---
> Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).
---

## Thought Process

There's the obvious linear time and constant space solution of just manually multiplying x n times in a loop. We want to do better and can easily achieve O(logN) when we use binary exponentiation.

Think about two basic properties of exponents:
1. x^a * x^b = x^(a + b)
2. (x^a)^b = x^(ab)

We can use these (especially Property 2) to drastically cut down on our calculations. Say we have a calculation like 10^100. That's actually the same as 10^100 = (10^2)^50 = 100^50. And we can repeat this over and over until our exponent becomes 0 or 1 (depending on our implementation. In this one, we continue until 0). 

We're dividing the exponent by 2, but what about when we have an odd number? That's where Property 1 comes in! For example, if we have x^25, that's equivalent to x^25 = (x^24) * (x^1). We can then continue our binary exponentiation on the even-numbered exponent and simply multiply the x value at the time into a result variable. The x value will continue changing, so that's why we want to perform this multiplication into the result at this exact point, while we still have a reference to the value.

Now we just have to quickly handle the edge cases at the start and make sure that our exponent is a positive value.

This can be done either recursively or iteratively. The iterative method is objectively better since we then only need constant O(1) space and don't need a recursive call stack. The time complexity for both would still be O(logN) since we're reducing the exponent N by half with each loop.

```
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0: return 1
        if n < 0: x = 1.0 / x; n *= -1

        product = 1
        while n:
            if n % 2: product *= x; n-= 1
            x *= x; n //= 2
        return product

        # Recursive solution that isn't used in this code
        def binary_exp(x: float, n: int) -> float:
            if n == 0: return 1
            if n < 0: return 1.0 / binary_exp(x, -n)

            if n % 2: return x * binary_exp(x * x, (n - 1) // 2)
            else: return binary_exp(x * x, n // 2)
```

# 43, Multiply Strings

## Problem Statement
---
> Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string. Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
---

## Thought Process

This hinges on the fact that the number of digits in the product between two numbers is either the total number of digits between the two numbers or 1 less than that. Similarly, when multiplying digits together, the final index is equal to the sum of the original indices of the two digits in the multiplicand and multiplier. 

Knowing this, the problem becomes trivial. We (optionally) want to first reverse the given input; this makes for a cleaner iteration and calculation of the final indices. If we don't reverse it, the index would have to be subtracted from the length of the input, or alternatively the loop would iterate in reverse.

When we determine the product of two digits and the place it should end up in, we only place the last digit in that index and sum the carry into the next index. The logic for this is intuitive if you do the multiplication by hand. The last step of basic multiplication involves adding those numbers together, so you're just doing this step earlier and index by index.

Finally, we should reformat the product as a string, making sure to re-reverse it if we did so at the start. Before that, we also want to check if the last digit (or the first digit if we didn't reverse at the start) exists; if it doesn't, we can pop it off. This goes back to the first principle we mentioned for the number of digits in the result, since we don't want leading zeroes.

The time complexity is O(MN), where M is the number of digits in one number and N is the number of digits in the other. This is because we just have to multiply and sum once each digit against all the other digits of th other number (worded it confusingly, but basic multiplication stuff).

The space complexity is O(M + N). While we usually ignore the space required for the result array, we don't compute the result string until the end, which we can't avoid because strings are immutable in Python. M + N is the max number of digits in the product, so we need to keep an array of that size until we're done and can convert into a string.

```
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0": return "0"

        first, second = num1[::-1], num2[::-1]
        product = [0] * (len(first) + len(second))

        for ind_1, num_1 in enumerate(first):
            for ind_2, num_2 in enumerate(second):
                ind = ind_1 + ind_2
                curr = int(num_1) * int(num_2) + product[ind]

                carry, product[ind] = divmod(curr, 10)
                product[ind + 1] += carry
        
        if not product[-1]: product.pop()
        return "".join([str(product[x]) for x in range(len(product) - 1, -1, -1)])
```

# 2013, Detect Squares

## Problem Statement
---
> You are given a stream of points on the X-Y plane. Design an algorithm that:
> Adds new points from the stream into a data structure. Duplicate points are allowed and should be treated as different points.
> Given a query point, counts the number of ways to choose three points from the data structure such that the three points and the query point form an axis-aligned square with positive area.
> An axis-aligned square is a square whose edges are all the same length and are either parallel or perpendicular to the x-axis and y-axis.
> Implement the DetectSquares class:
> DetectSquares() Initializes the object with an empty data structure.
> void add(int[] point) Adds a new point point = [x, y] to the data structure.
> int count(int[] point) Counts the number of ways to form axis-aligned squares with point point = [x, y] as described above.
---

## Thought Process

Keeping this super short, but we're storing the points in a dictionary within a dictionary. That means for our Count function, we can easily access all the points on the same vertical line (same x-value). We can compute different side lengths based on the difference in y-valus compared to our input, and we can thus check for points that are at x +/- height. 

We check for both of these situations, since it could be one, the other, or both! If there's a single point that doesn't exist, it would return 0, and since we're multiplying, that would make it so that we're adding 0. That's how we're handling if the points we're looking for don't exist; it's basically handling itself!

Remember that the dictionary isn't counted for our space usage, since we're creating it during initialization. The time and space complexity for the Add function is O(1); we're just accessing a value in a dictionary and incrementing it. For the Count, the time complexity is O(N) in the worst case, where N is all the points we've stored; realistically it's just the number of points in the dictionary for the given x-value. The space complexity is O(1); we're not doing any scaled storage here. 

If we consider the class as a whole, the space complexity would be O(N) however.

```
class DetectSquares:
    def __init__(self):
        self.points = collections.defaultdict(collections.Counter)        

    def add(self, point: List[int]) -> None:
        self.points[point[0]][point[1]] += 1
        
    def count(self, point: List[int]) -> int:
        ans = 0
        x1, y1 = point

        for y2 in self.points[x1]:
            if y2 == y1: continue
            length = abs(y2 - y1)
            ans += self.points[x1][y2] * self.points[x1 + length][y1] * self.points[x1 + length][y2]
            ans += self.points[x1][y2] * self.points[x1 - length][y1] * self.points[x1 - length][y2]

        return ans
```

