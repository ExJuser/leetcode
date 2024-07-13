package main

import (
	"slices"
	"sort"
	"strconv"
	"strings"
)

//	func findLeftAndRight(height []int, value int) []int {
//		left := 0
//		right := len(height) - 1
//		for ; left < right; left++ {
//			if height[left] >= value {
//				break
//			}
//		}
//		for ; left < right; right-- {
//			if height[right] >= value {
//				break
//			}
//		}
//		if left != right {
//			return []int{left, right}
//		} else {
//			return []int{-1}
//		}
//	}
//
//	func maxArea(height []int) int {
//		m := 0
//		for i := 1; ; i++ {
//			leftAndRight := findLeftAndRight(height, i)
//			if leftAndRight[0] == -1 {
//				break
//			} else {
//				area := (leftAndRight[1] - leftAndRight[0]) * i
//				m = max(m, area)
//			}
//		}
//		return m
//	}

func maxArea(height []int) int {
	left := 0
	right := len(height) - 1
	m := 0
	for left < right {
		area := (right - left) * min(height[left], height[right])
		m = max(m, area)
		if height[left] <= height[right] {
			left++
		} else {
			right--
		}
	}
	return m
}
func moveZeroes(nums []int) {
	slow := 0
	for fast := 0; fast < len(nums); fast++ {
		if nums[fast] != 0 {
			nums[slow] = nums[fast]
			slow++
		}
	}
	for ; slow < len(nums); slow++ {
		nums[slow] = 0
	}
}
func isIsomorphic(s string, t string) bool {
	if len(s) != len(s) {
		return false
	} else {
		mapping1 := make(map[string]string)
		mapping2 := make(map[string]string)
		for i := 0; i < len(s); i++ {
			if x, ok := mapping1[s[i:i+1]]; ok {
				if x != t[i:i+1] {
					return false
				}
			} else {
				if _, ok = mapping2[t[i:i+1]]; !ok {
					mapping1[s[i:i+1]] = t[i : i+1]
					mapping2[t[i:i+1]] = s[i : i+1]
				} else {
					return false
				}
			}
		}
		return true
	}
}
func longestConsecutive(nums []int) int {
	if len(nums) == 0 {
		return 0
	} else {
		set := make(map[int]bool)
		for _, num := range nums {
			if _, ok := set[num]; !ok {
				set[num] = true
			}
		}
		maxLen := 0
		for k, v := range set {
			if v {
				set[k] = false
				up, down := k+1, k-1
				_, ok := set[up]
				for ok {
					set[up] = false
					up++
					_, ok = set[up]
				}
				_, ok = set[down]
				for ok {
					set[down] = false
					down--
					_, ok = set[down]
				}
				maxLen = max(maxLen, up-down-1)
			}
		}
		return maxLen
	}
}

//	func reverseString(s []byte) {
//		length := len(s)
//		if length == 1 {
//			return
//		}
//		for i := 0; i < length/2; i++ {
//			s[i], s[length-i-1] = s[length-i-1], s[i]
//		}
//	}
func reverseStr(s string, k int) string {
	length := len(s)
	sb := []byte(s)
	for p := 0; p < length; p += 2 * k {
		if length-p < k {
			slices.Reverse(sb[p:])
		} else {
			slices.Reverse(sb[p : p+k])
		}
	}
	return string(sb)
}

func reverseWords(s string) string {
	words := make([]string, 0, 10)
	for i := 0; i < len(s); {
		if s[i] != ' ' {
			j := i
			for ; j < len(s); j++ {
				if s[j] == ' ' {
					break
				}
			}
			words = append(words, s[i:j])
			i = j
		} else {
			i++
		}
	}
	slices.Reverse(words)
	str := strings.Builder{}
	for i := 0; i < len(words); i++ {
		if i != len(words)-1 {
			str.WriteString(words[i] + " ")
		} else {
			str.WriteString(words[i])
		}
	}
	return str.String()
}

//
// "the sky is blue"
//func reverseWords(s string) string {
//	length := len(s)
//	builder := strings.Builder{}
//	for i := length - 1; i >= 0; {
//		if s[i] != ' ' {
//			j := i
//			for ; j >= 0; j-- {
//				if s[j] == ' ' {
//					break
//				}
//			}
//			builder.WriteString(s[j+1:i+1] + " ")
//			i = j
//		} else {
//			i--
//		}
//	}
//	ans := builder.String()
//	return ans[:len(ans)-1]
//}

//	func strStr(haystack string, needle string) int {
//		mainString := []byte(haystack)
//		substring := []byte(needle)
//		if len(substring) > len(mainString) {
//			return -1
//		} else {
//			next := getNext(substring)
//			i, j := 0, 0
//			for i < len(mainString) && j < len(substring) {
//				if mainString[i] == substring[j] {
//					i, j = i+1, j+1
//				} else if j > 0 {
//					j = next[j]
//				} else {
//					i++
//				}
//			}
//			if j == len(substring) {
//				return i - j
//			} else {
//				return -1
//			}
//		}
//	}
func buildArray(target []int, n int) (ans []string) {
	k := 1
	for i := 0; i < len(target); i++ {
		if target[i] == k {
			ans = append(ans, "Push")
			k++
		} else {
			for target[i] != k {
				ans = append(ans, "Push", "Pop")
				k++
			}
			ans = append(ans, "Push")
			k++
		}
	}
	return
}
func finalValueAfterOperations(operations []string) (ans int) {
	for _, op := range operations {
		if op == "++X" || op == "X++" {
			ans += 1
		} else {
			ans -= 1
		}
	}
	return
}
func maxNumberOfBalloons(text string) int {
	balloonCount := make(map[string]int)
	for i := 0; i < len(text); i++ {
		if strings.Contains("balloon", text[i:i+1]) {
			balloonCount[text[i:i+1]] += 1
		}
	}
	count := len(text)
	if len(balloonCount) != 5 {
		return 0
	}
	for k, v := range balloonCount {
		if strings.Contains("abn", k) {
			count = min(count, v)
		} else if strings.Contains("lo", k) {
			count = min(count, v/2)
		}
	}
	return count
}

//	func countPoints(rings string) (count int) {
//		if len(rings) < 6 {
//			return 0
//		}
//		ringMap := make(map[int]string)
//		for i := 0; i < len(rings)/2; i++ {
//			ringMap[int(rings[2*i+1])] += rings[2*i : 2*i+1]
//		}
//		for _, v := range ringMap {
//			if strings.Contains(v, "R") && strings.Contains(v, "G") && strings.Contains(v, "B") {
//				count++
//			}
//		}
//		return
//	}
func canBeEqual(target []int, arr []int) bool {
	slices.Sort(arr)
	slices.Sort(target)
	return slices.Equal(target, arr)
}
func sumOfSquares(nums []int) (ans int) {
	n := len(nums)
	for i, num := range nums {
		if n%(i+1) == 0 {
			ans += num * num
		}
	}
	return
}
func getConcatenation(nums []int) []int {
	return append(nums, nums...)
}
func countNegatives(grid [][]int) (count int) {
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] < 0 {
				count++
			}
		}
	}
	return
}
func findNumbers(nums []int) (count int) {
	for _, num := range nums {
		if len(strconv.Itoa(num))%2 == 0 {
			count++
		}
	}
	return
}
func rangeSum(nums []int, n int, left int, right int) (ans int) {
	numsSum := make([]int, 0, n*(n+1)/2)
	for i := 0; i < n; i++ {
		sum := 0
		for j := i; j < n; j++ {
			sum += nums[j]
			numsSum = append(numsSum, sum)
		}
	}
	slices.Sort(numsSum)
	for i := left - 1; i < right; i++ {
		ans += numsSum[i]
	}
	ans %= 1000000007
	return
}
func countStudents(students []int, sandwiches []int) int {
	for len(sandwiches) > 0 {
		if students[0] != sandwiches[0] {
			if slices.Contains(students, sandwiches[0]) {
				students = append(students[1:], students[0])
			} else {
				break
			}
		} else {
			students = students[1:]
			sandwiches = sandwiches[1:]
		}
	}
	return len(students)
}
func findSolution(customFunction func(int, int) int, z int) (ans [][]int) {
	for i := 1; i <= 1000; i++ {
		for j := 1; j <= 1000; j++ {
			res := customFunction(i, j)
			if res == z {
				ans = append(ans, []int{i, j})
			} else if res > z {
				break
			}
		}
	}
	return
}
func maxCoins(piles []int) (ans int) {
	n := len(piles)
	slices.Sort(piles)
	takeCount := n / 3
	for i := 0; i < takeCount; i++ {
		ans += piles[n-(i+1)*2]
	}
	return ans
}
func canVisitAllRooms(rooms [][]int) bool {
	n := len(rooms)
	visited := make([]bool, n)
	keys := make([]int, 0, n)
	visited[0] = true
	for _, key := range rooms[0] {
		keys = append(keys, key)
	}
	for len(keys) > 0 {
		if !visited[keys[0]] {
			visited[keys[0]] = true
			for _, num := range rooms[keys[0]] {
				keys = append(keys, num)
			}
		} else {
			keys = keys[1:]
		}
	}
	for _, ok := range visited {
		if !ok {
			return false
		}
	}
	return true
}
func canBeTypedWords(text string, brokenLetters string) (count int) {
	wordSlice := strings.Split(text, " ")
	for _, word := range wordSlice {
		for _, char := range brokenLetters {
			if strings.Contains(word, string(char)) {
				count++
				break
			}
		}
	}
	count = len(wordSlice) - count
	return
}
func truncateSentence(s string, k int) (x string) {
	for i, char := range s {
		if char == ' ' {
			k--
		}
		if k == 0 {
			x = s[:i]
			break
		}
		if len(s)-1 == i {
			x = s[:i+1]
			break
		}
	}
	return
}
func checkZeroOnes(s string) bool {
	countMap := make(map[byte]int)
	left, right := 0, 0
	for left < len(s) && right < len(s) {
		if s[right] != s[left] {
			countMap[s[left]] = max(countMap[s[left]], right-left)
			left = right
		}
		if right == len(s)-1 {
			countMap[s[left]] = max(countMap[s[left]], right-left+1)
		}
		right++
	}
	return countMap['1'] > countMap['0']
}
func countCharacters(words []string, chars string) (count int) {
	charMap := make(map[byte]int)
	wordMap := make(map[byte]int)
	for _, char := range chars {
		charMap[byte(char)] += 1
	}
	for _, word := range words {
		clear(wordMap)
		for _, char := range word {
			wordMap[byte(char)] += 1
		}
		flag := true
		for k, v := range wordMap {
			if v > charMap[k] {
				flag = false
				break
			}
		}
		if flag {
			count += len(word)
		}
	}
	return
}

func leftRightDifference(nums []int) []int {
	sumDiff := make([]int, len(nums))
	var sumCenter, sumLeft, sumRight, sumTotal int
	for _, num := range nums {
		sumTotal += num
	}
	for i, num := range nums {
		if i == 0 {
			sumLeft = 0
		} else {
			sumLeft = sumCenter
		}
		sumCenter += num
		sumRight = sumTotal - sumCenter
		sumDiff[i] = Abs(sumLeft - sumRight)
	}
	return sumDiff
}
func oddString(words []string) string {
	var formerDifference []byte
	flagIndex := -1
	for i, word := range words {
		difference := make([]byte, len(word)-1)
		for i := 0; i < len(word)-1; i++ {
			difference[i] = word[i+1] - word[i]
		}
		if i > 0 && !slices.Equal(formerDifference, difference) {
			if flagIndex != -1 {
				return words[flagIndex]
			} else {
				flagIndex = i
			}
		}
		formerDifference = difference
	}
	if flagIndex == len(words)-1 {
		return words[flagIndex]
	}
	return words[0]
}

//	func reverseString(s string) string {
//		byteArr := []byte(s)
//		slices.Reverse(byteArr)
//		return string(byteArr)
//	}
func maximumNumberOfStringPairs(words []string) (count int) {
	visited := make(map[int]bool)
	for i := 0; i < len(words); i++ {
		if visited[i] {
			continue
		}
		for j := i + 1; j < len(words); j++ {
			if visited[j] {
				continue
			}
			if reverseString(words[j]) == words[i] {
				visited[i] = true
				visited[j] = true
				count++
			}
		}
	}
	return
}
func smallestString(s string) string {
	byteArr := []byte(s)
	left := 0
	for left < len(s) && s[left] == 'a' {
		left++
	}
	right := left
	for right < len(s) && s[right] != 'a' {
		right++
	}
	if left == len(s) {
		byteArr[len(s)-1] = 'z'
	} else {
		for i := left; i < right; i++ {
			byteArr[i] -= 1
		}
	}
	return string(byteArr)
}
func subtractProductAndSum(n int) int {
	prod := 1
	sum := 0
	for n > 0 {
		digit := n % 10
		prod *= digit
		sum += digit
		n /= 10
	}
	return prod - sum
}
func tribonacci(n int) int {
	tribonacciSlice := make([]int, 38)
	tribonacciSlice[0] = 0
	tribonacciSlice[1] = 1
	tribonacciSlice[2] = 1
	if n == 0 {
		return 0
	} else if n == 1 {
		return 1
	} else if n == 2 {
		return 1
	} else {
		for i := 3; i <= n; i++ {
			tribonacciSlice[i] = tribonacciSlice[i-1] + tribonacciSlice[i-2] + tribonacciSlice[i-3]
		}
	}
	return tribonacciSlice[n]
}

//	func maxProductDifference(nums []int) int {
//		slices.Sort(nums)
//		return nums[len(nums)-1]*nums[len(nums)-2] - nums[0]*nums[1]
//	}

func smallestEvenMultiple(n int) int {
	if n%2 == 0 {
		return n
	} else {
		return n * 2
	}
}

//	func groupAnagrams(strs []string) (ans [][]string) {
//		if len(strs) == 0 {
//			return
//		}
//		if len(strs) == 1 {
//			ans = append(ans, strs)
//			return
//		}
//		anagramsMap := make(map[string][]string)
//		for _, str := range strs {
//			bytes := []byte(str)
//			slices.Sort(bytes)
//			anagramsMap[string(bytes)] = append(anagramsMap[string(bytes)], str)
//		}
//		for _, v := range anagramsMap {
//			ans = append(ans, v)
//		}
//		return
//	}
func findAnagrams(s string, p string) (ans []int) {
	if len(p) > len(s) {
		return
	}
	length := len(p)
	countP := make(map[byte]int)
	countS := make(map[byte]int)
	for _, char := range p {
		countP[byte(char)] += 1
	}
	for i := 0; i < length; i++ {
		countS[s[i]] += 1
	}
	flag := true
	for k, v := range countS {
		if v != countP[k] {
			flag = false
			break
		}
	}
	if flag {
		ans = append(ans, 0)
	}
	for i := 1; i <= len(s)-length; i++ {
		countS[s[i-1]] -= 1
		countS[s[i+length-1]] += 1
		if countS[s[i-1]] != countP[s[i-1]] || countS[s[i+length-1]] != countP[s[i+length-1]] {
			continue
		} else {
			flag = true
			for k, v := range countS {
				if v != countP[k] {
					flag = false
					break
				}
			}
			if flag {
				ans = append(ans, i)
			}
		}
	}
	return
}

//	func subarraySum(nums []int, k int) (count int) {
//		length := len(nums)
//		center := make([]int, length)
//		left := make([]int, length)
//		sum := 0
//		for i := 0; i < length; i++ {
//			left[i] = sum
//			sum += nums[i]
//			center[i] = sum
//		}
//		for i := 0; i < length; i++ {
//			for j := i; j < length; j++ {
//				if center[j]-left[i] == k {
//					count++
//				}
//			}
//		}
//		return
//	}
func subarraySum(nums []int, k int) (ans int) {
	prefix := make(map[int]int)
	prefix[0] = 1
	sum := 0
	for _, num := range nums {
		sum += num
		prefix[sum] += 1
		if count, ok := prefix[sum-k]; ok {
			ans += count
		}
	}
	return
}

//	func merge(intervals [][]int) (ans [][]int) {
//		slices.SortFunc(intervals, func(a, b []int) int {
//			return a[0] - b[0]
//		})
//		temp := make([]int, 0)
//		for _, interval := range intervals {
//			if len(temp) == 0 {
//				temp = interval
//			} else {
//				if interval[0] > temp[1] {
//					ans = append(ans, temp)
//					temp = interval
//				} else if temp[1] < interval[1] {
//					temp[1] = interval[1]
//				}
//			}
//		}
//		if len(temp) != 0 {
//			ans = append(ans, temp)
//		}
//		return
//	}
func productExceptSelf(nums []int) (ans []int) {
	length := len(nums)
	prefixProd := make([]int, length)
	suffixProd := make([]int, length)
	prod := 1
	for i := 0; i < length; i++ {
		prefixProd[i] = prod
		prod *= nums[i]
	}
	prod = 1
	for i := length - 1; i >= 0; i-- {
		suffixProd[i] = prod
		prod *= nums[i]
	}
	for i := 0; i < length; i++ {
		ans = append(ans, prefixProd[i]*suffixProd[i])
	}
	return
}
func setZeroes(matrix [][]int) {
	zeroRowSet := make(map[int]struct{})
	zeroColSet := make(map[int]struct{})
	for i, row := range matrix {
		for j, num := range row {
			if num == 0 {
				zeroRowSet[i] = struct{}{}
				zeroColSet[j] = struct{}{}
			}
		}
	}
	for k, _ := range zeroRowSet {
		clear(matrix[k])
	}
	for i, row := range matrix {
		if _, ok := zeroRowSet[i]; !ok {
			for j, _ := range row {
				if _, ok := zeroColSet[j]; ok {
					row[j] = 0
				}
			}
		}
	}
}

//	func getIntersectionNode(headA, headB *ListNode) *ListNode {
//		var lenA, lenB int
//		pA, pB := headA, headB
//		for pA != nil {
//			lenA++
//			pA = pA.Next
//		}
//		for pB != nil {
//			lenB++
//			pB = pB.Next
//		}
//		fmt.Println(lenA, lenB)
//		pA, pB = headA, headB
//		if lenA > lenB {
//			delta := lenA - lenB
//			for ; pA != nil && delta > 0; delta-- {
//				pA = pA.Next
//			}
//		} else {
//			delta := lenB - lenA
//			for ; pB != nil && delta > 0; delta-- {
//				pB = pB.Next
//			}
//		}
//		for pA != pB && pA != nil && pB != nil {
//			pA = pA.Next
//			pB = pB.Next
//		}
//		return pA
//	}
func findNonMinOrMax(nums []int) int {
	slices.Sort(nums)
	if len(nums) <= 2 {
		return -1
	} else {
		return nums[1]
	}
}
func averageValue(nums []int) int {
	var sum, count int
	for _, num := range nums {
		if num%6 == 0 {
			sum += num
			count++
		}
	}
	if count == 0 {
		return 0
	} else {
		return sum / count
	}
}

//	func getDecimalValue(head *ListNode) int {
//		strBuilder := strings.Builder{}
//		for head != nil {
//			strBuilder.WriteString(strconv.Itoa(head.Val))
//			head = head.Next
//		}
//		res, _ := strconv.ParseInt(strBuilder.String(), 2, 64)
//		return int(res)
//	}
func smallerNumbersThanCurrent(nums []int) []int {
	newNums := make([]int, len(nums))
	countMap := make(map[int]int)
	copy(newNums, nums)
	slices.Sort(newNums)
	for _, num := range newNums {
		if _, ok := countMap[num]; !ok {
			countMap[num] = sort.Search(len(newNums), func(i int) bool {
				return newNums[i] >= num
			})
		}
	}
	ans := make([]int, len(nums))
	for i, num := range nums {
		ans[i] = countMap[num]
	}
	return ans
}
func targetIndices(nums []int, target int) (ans []int) {
	slices.Sort(nums)
	for i, num := range nums {
		if num == target {
			ans = append(ans, i)
		}
	}
	return
}
func convertTemperature(celsius float64) []float64 {
	return []float64{celsius + 273.15, 1.80*celsius + 32.00}
}
func canMakeArithmeticProgression(arr []int) bool {
	if len(arr) == 2 {
		return true
	} else {
		slices.Sort(arr)
		delta := arr[1] - arr[0]
		for i := 2; i < len(arr); i++ {
			if arr[i]-arr[i-1] != delta {
				return false
			}
		}
	}
	return true
}

//	func dailyTemperatures(temperatures []int) []int {
//		stack := make([]int, 0, 10)
//		ans := make([]int, len(temperatures))
//		for i, t := range temperatures {
//			if len(stack) == 0 || temperatures[stack[len(stack)-1]] >= t {
//				stack = append(stack, i)
//			} else if temperatures[stack[len(stack)-1]] < t {
//				for len(stack) > 0 && temperatures[stack[len(stack)-1]] < t {
//					ans[stack[len(stack)-1]] = i - stack[len(stack)-1]
//					stack = stack[:len(stack)-1]
//				}
//				stack = append(stack, i)
//			}
//		}
//		return ans
//	}
func nextGreaterElement(nums1 []int, nums2 []int) (ans []int) {
	stack := make([]int, 0, 10)
	hash := make(map[int]int)
	for i, num := range nums2 {
		if len(stack) == 0 || num <= nums2[stack[len(stack)-1]] {
			stack = append(stack, i)
		} else if num > nums2[stack[len(stack)-1]] {
			for len(stack) > 0 && nums2[stack[len(stack)-1]] < num {
				hash[nums2[stack[len(stack)-1]]] = num
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, i)
		}
	}
	for _, num := range nums1 {
		if hash[num] != 0 {
			ans = append(ans, hash[num])
		} else {
			ans = append(ans, -1)
		}
	}
	return
}

//	func removeDuplicateLetters(s string) string {
//		stack := make([]byte, 0, 10)
//		countMap := make(map[byte]int)
//		for _, char := range s {
//			countMap[byte(char)] += 1
//		}
//		for _, char := range s {
//			if !slices.Contains(stack, byte(char)) {
//				if len(stack) == 0 || byte(char) > stack[len(stack)-1] {
//					stack = append(stack, byte(char))
//				} else if byte(char) < stack[len(stack)-1] {
//					for len(stack) > 0 && byte(char) <= stack[len(stack)-1] {
//						if countMap[stack[len(stack)-1]] > 0 {
//							stack = stack[:len(stack)-1]
//						} else {
//							break
//						}
//					}
//					stack = append(stack, byte(char))
//				}
//			}
//			countMap[byte(char)] -= 1
//		}
//		return string(stack)
//	}
func removeKdigits(num string, k int) string {
	stack := make([]byte, 0, 10)
	for _, digit := range num {
		if len(stack) == 0 || byte(digit) >= stack[len(stack)-1] {
			stack = append(stack, byte(digit))
		} else if byte(digit) < stack[len(stack)-1] {
			for k > 0 && len(stack) > 0 && byte(digit) < stack[len(stack)-1] {
				stack = stack[:len(stack)-1]
				k--
			}
			stack = append(stack, byte(digit))
		}
	}
	for k > 0 {
		stack = stack[:len(stack)-1]
		k--
	}
	ans := strings.TrimLeft(string(stack), "0")
	if ans == "" {
		return "0"
	} else {
		return ans
	}
}
func nextLargerNodes(head *ListNode) []int {
	nums := make([]int, 0)
	stack := make([]int, 0)
	for head != nil {
		nums = append(nums, head.Val)
		head = head.Next
	}
	ans := make([]int, len(nums))
	for i, num := range nums {
		if len(stack) == 0 || num <= nums[stack[len(stack)-1]] {
			stack = append(stack, i)
		} else if num > nums[stack[len(stack)-1]] {
			for len(stack) > 0 && num > nums[stack[len(stack)-1]] {
				ans[stack[len(stack)-1]] = num
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, i)
		}
	}
	return ans
}
