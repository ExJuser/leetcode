package main

import (
	"fmt"
	"math"
	"math/bits"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

func findKDistantIndices(nums []int, key int, k int) (ans []int) {
	for i := 0; i < len(nums); i++ {
		if nums[i] == key {
			var j int
			if len(ans) != 0 && i-k <= ans[len(ans)-1] {
				j = ans[len(ans)-1] + 1
			} else {
				j = max(0, i-k)
			}
			for ; j <= i+k && j < len(nums); j++ {
				ans = append(ans, j)
			}
		}
	}
	return
}

func distinctDifferenceArray(nums []int) []int {
	prefixMap := make(map[int]struct{})
	prefixArr := make([]int, len(nums))
	prefix := 0
	suffixMap := make(map[int]struct{})
	suffixArr := make([]int, len(nums))
	suffix := 0
	for i := 0; i < len(nums); i++ {
		if _, ok := prefixMap[nums[i]]; !ok {
			prefix++
			prefixMap[nums[i]] = struct{}{}
		}
		prefixArr[i] = prefix
	}
	for i := len(nums) - 1; i >= 0; i-- {
		suffixArr[i] = suffix
		if _, ok := suffixMap[nums[i]]; !ok {
			suffix++
			suffixMap[nums[i]] = struct{}{}
		}
	}
	ans := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		ans[i] = prefixArr[i] - suffixArr[i]
	}
	return ans
}

func hardestWorker(n int, logs [][]int) int {
	var longest, longestID int
	for i, log := range logs {
		if i > 0 {
			if log[1]-logs[i-1][1] > longest {
				longest = log[1] - logs[i-1][1]
				longestID = log[0]
			} else if log[1]-logs[i-1][1] == longest {
				longestID = min(longestID, log[0])
			}
		} else {
			longest = log[1]
			longestID = log[0]
		}
	}
	return longestID
}

func groupThePeople(groupSizes []int) (ans [][]int) {
	isInGroup := make([]bool, len(groupSizes))
	for i := 0; i < len(groupSizes); i++ {
		if !isInGroup[i] {
			size := groupSizes[i]
			group := make([]int, 0, size)
			for j := 0; j < len(groupSizes); j++ {
				if !isInGroup[j] && groupSizes[j] == size {
					if len(group) != size {
						group = append(group, j)
						isInGroup[j] = true
					} else {
						break
					}
				}
			}
			ans = append(ans, group)
		}
	}
	return
}

func decodeMessage(key string, message string) string {
	coding := make(map[byte]byte)
	var index byte
	for _, char := range key {
		if char != ' ' {
			if len(coding) >= 26 {
				break
			}
			if _, ok := coding[byte(char)]; !ok {
				coding[byte(char)] = 'a' + index
				index++
			}
		}
	}
	builder := strings.Builder{}
	for _, char := range message {
		if char == ' ' {
			builder.WriteByte(' ')
		} else {
			builder.WriteByte(coding[byte(char)])
		}
	}
	return builder.String()
}

//	func simplifiedFractions(n int) (ans []string) {
//		for j := 2; j <= n; j++ {
//			for k := 1; k < j; k++ {
//				if findGCD([]int{j, k}) == 1 {
//					ans = append(ans, fmt.Sprintf("%d/%d", k, j))
//				}
//			}
//		}
//		return
//	}

func twoOutOfThree(nums1 []int, nums2 []int, nums3 []int) (ans []int) {
	set1 := make(map[int]struct{})
	set2 := make(map[int]struct{})
	set3 := make(map[int]struct{})
	countMap := make(map[int]int)
	for _, num := range nums1 {
		if _, ok := set1[num]; !ok {
			set1[num] = struct{}{}
		}
	}
	for k, _ := range set1 {
		countMap[k]++
	}
	for _, num := range nums2 {
		if _, ok := set2[num]; !ok {
			set2[num] = struct{}{}
		}
	}
	for k, _ := range set2 {
		countMap[k]++
	}
	for _, num := range nums3 {
		if _, ok := set3[num]; !ok {
			set3[num] = struct{}{}
		}
	}
	for k, _ := range set3 {
		countMap[k]++
	}
	for k, v := range countMap {
		if v >= 2 {
			ans = append(ans, k)
		}
	}
	return
}

func mergeSimilarItems(items1 [][]int, items2 [][]int) (ans [][]int) {
	itemMap := make(map[int]int)
	for _, item := range items1 {
		itemMap[item[0]] = item[1]
	}
	for _, item := range items2 {
		itemMap[item[0]] += item[1]
	}
	for k, v := range itemMap {
		ans = append(ans, []int{k, v})
	}
	slices.SortFunc(ans, func(a, b []int) int {
		return a[0] - b[0]
	})
	return
}

func isLongPressedName(name string, typed string) bool {
	var i, j int
	var prevChar byte

	// 比较name和typed的每个字符
	for i < len(name) && j < len(typed) {
		if name[i] == typed[j] {
			// 记录当前匹配的字符
			prevChar = name[i]
			i++
			j++
		} else {
			// 检查是否为长按键入的情况
			if typed[j] != prevChar {
				return false
			}

			// 跳过连续的相同字符
			for j < len(typed) && typed[j] == prevChar {
				j++
			}
		}
	}

	// 检查是否name中的字符已经比较完
	if i < len(name) {
		return false
	} else {
		// 跳过typed中剩余的相同字符
		for j < len(typed) && typed[j] == prevChar {
			j++
		}
		// 检查是否typed中的字符已经比较完
		return j == len(typed)
	}
}

func thousandSeparator(n int) string {
	numStr := strconv.Itoa(n)
	length := len(numStr)
	for i := length - 1; i > 2; i -= 3 {
		numStr = numStr[:i-2] + "." + numStr[i-2:]
	}
	return numStr
}

//	func countKDifference(nums []int, k int) (count int) {
//		for i := 0; i < len(nums); i++ {
//			for j := i + 1; j < len(nums); j++ {
//				if Abs(nums[i]-nums[j]) == k {
//					count++
//				}
//			}
//		}
//		return count
//	}

func divideString(s string, k int, fill byte) (ans []string) {
	i := 0
	for ; i+k < len(s); i += k {
		ans = append(ans, s[i:i+k])
	}
	builder := strings.Builder{}
	builder.WriteString(s[i:])
	for k-builder.Len() > 0 {
		builder.WriteByte(fill)
	}
	ans = append(ans, builder.String())
	return
}

func checkAlmostEquivalent(word1 string, word2 string) bool {
	count := make(map[byte]int)
	for _, char := range word1 {
		count[byte(char)]++
	}
	for _, char := range word2 {
		count[byte(char)]--
	}
	for _, v := range count {
		if v > 3 || v < -3 {
			return false
		}
	}
	return true
}

func largestSumAfterKNegations(nums []int, k int) int {
	slices.Sort(nums)
	var i, sum int
	for k > 0 {
		if i == len(nums) || nums[i] > 0 {
			if k%2 != 0 {
				nums[0] -= 2 * slices.Min(nums)
			}
			k = 0
		} else if nums[i] < 0 {
			nums[i] *= -1
			k--
			i++
		} else if nums[i] == 0 {
			k = 0
		}
	}
	for _, num := range nums {
		sum += num
	}
	return sum
}

func findOcurrences(text string, first string, second string) (ans []string) {
	words := strings.Split(text, " ")
	for i := 0; i < len(words)-2; i++ {
		if words[i] == first && words[i+1] == second {
			ans = append(ans, words[i+2])
		}
	}
	return
}

func capitalizeTitle(title string) string {
	words := strings.Split(strings.ToLower(title), " ")
	builder := strings.Builder{}
	for i := 0; i < len(words); i++ {
		if len(words[i]) <= 2 {
			builder.WriteString(words[i])
		} else {
			builder.WriteString(strings.ToUpper(words[i][0:1]) + words[i][1:])
		}
		if i != len(words)-1 {
			builder.WriteString(" ")
		}
	}
	return builder.String()
}

func findLonely(nums []int) (ans []int) {
	count := make(map[int]int)
	for _, num := range nums {
		count[num]++
	}
	for k, v := range count {
		if v == 1 && count[k-1] == 0 && count[k+1] == 0 {
			ans = append(ans, k)
		}
	}
	return
}

func noZero(n int) bool {
	for n > 0 {
		if n%10 == 0 {
			return false
		}
		n /= 10
	}
	return true
}

func getNoZeroIntegers(n int) []int {
	for i := 1; i <= n/2; i++ {
		if noZero(i) && noZero(n-i) {
			return []int{i, n - i}
		}
	}
	return []int{}
}

//	func digitSum(n int) (sum int) {
//		for n > 0 {
//			sum += n % 10
//			n /= 10
//		}
//		return
//	}
//
//	func countBalls(lowLimit int, highLimit int) int {
//		boxes := make(map[int]int)
//		for i := lowLimit; i <= highLimit; i++ {
//			boxes[digitSum(i)]++
//		}
//		maxBall := math.MinInt
//		for _, v := range boxes {
//			maxBall = max(maxBall, v)
//		}
//		return maxBall
//	}

func passThePillow(n int, time int) int {
	i, k := 1, 1
	for time > 0 {
		if i == 1 {
			k = 1
		} else if i == n {
			k = -1
		}
		i += k
		time--
	}
	return i
}

//func countGoodTriplets(arr []int, a int, b int, c int) (count int) {
//	for i := 0; i < len(arr); i++ {
//		for j := i + 1; j < len(arr); j++ {
//			for k := j + 1; k < len(arr); k++ {
//				if Abs(arr[i]-arr[j]) <= a && Abs(arr[j]-arr[k]) <= b && Abs(arr[i]-arr[k]) <= c {
//					count++
//				}
//			}
//		}
//	}
//	return
//}

func commonChars(words []string) (ans []string) {
	common := make([]int, 26)
	for i, word := range words {
		count := make([]int, 26)
		for _, char := range word {
			count[char-'a']++
		}
		if i == 0 {
			common = count
		} else {
			for i := 0; i < len(common); i++ {
				common[i] = min(common[i], count[i])
			}
		}
	}
	for i, times := range common {
		for j := 0; j < times; j++ {
			ans = append(ans, string(rune(i+'a')))
		}
	}
	return
}

func divisorSubstrings(num int, k int) (count int) {
	numStr := strconv.Itoa(num)
	length := len(numStr)
	for i := 0; i <= length-k; i++ {
		val, _ := strconv.Atoi(numStr[i : i+k])
		if val != 0 && num%val == 0 {
			count++
		}
	}
	return
}

func numberOfBeams(bank []string) (ans int) {
	countPrev := 0
	for i := 0; i < len(bank); i++ {
		count := strings.Count(bank[i], "1")
		if count != 0 {
			ans += countPrev * count
			countPrev = count
		}
	}
	return
}

func mergeArrays(nums1 [][]int, nums2 [][]int) (ans [][]int) {
	count := make(map[int]int)
	for i := 0; i < len(nums1); i++ {
		count[nums1[i][0]] += nums1[i][1]
	}
	for i := 0; i < len(nums2); i++ {
		count[nums2[i][0]] += nums2[i][1]
	}
	for k, v := range count {
		ans = append(ans, []int{k, v})
	}
	slices.SortFunc(ans, func(a, b []int) int {
		return a[0] - b[0]
	})
	return
}

func maxLengthBetweenEqualCharacters(s string) int {
	hash := make(map[byte]int)
	maxDist := -1
	for i, char := range s {
		if _, ok := hash[byte(char)]; ok {
			maxDist = max(maxDist, i-hash[byte(char)]-1)
		} else {
			hash[byte(char)] = i
		}
	}
	return maxDist
}

//	func minBitFlips(start int, goal int) (count int) {
//		for start > 0 || goal > 0 {
//			if start&1 != goal&1 {
//				count++
//			}
//			start >>= 1
//			goal >>= 1
//		}
//		return
//	}

func minBitFlips(start int, goal int) int {
	return bits.OnesCount(uint(start ^ goal))
}

func minLength(s string) int {
	for strings.Contains(s, "AB") || strings.Contains(s, "CD") {
		before, after, _ := strings.Cut(s, "AB")
		s = before + after
		before, after, _ = strings.Cut(s, "CD")
		s = before + after
	}
	return len(s)
}

//	func sumBase(n int, k int) (ans int) {
//		for n/k >= k {
//			ans += n % k
//			n = n / k
//		}
//		ans += n%k + n/k
//		return
//	}

func sumBase(n int, k int) (ans int) {
	for n > 0 {
		ans += n % k
		n = n / k
	}
	return
}

func lengthOfNum(n int) (length int) {
	if n < 0 {
		length++
	}
	for n != 0 {
		length++
		n /= 10
	}
	return
}

func findColumnWidth(grid [][]int) []int {
	maxLen := make([]int, len(grid[0]))
	for i := 0; i < len(maxLen); i++ {
		maxLen[i] = 1
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if lengthOfNum(grid[i][j]) > maxLen[j] {
				maxLen[j] = lengthOfNum(grid[i][j])
			}
		}
	}
	return maxLen
}

func isAcronym(words []string, s string) bool {
	if len(words) != len(s) {
		return false
	}
	builder := strings.Builder{}
	for _, word := range words {
		builder.WriteByte(word[0])
	}
	return builder.String() == s
}

func findContentChildren(g []int, s []int) int {
	slices.Sort(g)
	slices.Sort(s)
	var count, j int
	for i := 0; i < len(g); i++ {
		for j < len(s) && s[j] < g[i] {
			j++
		}
		if j < len(s) {
			count++
			j++
		}
	}
	return count
}

func wiggleMaxLength(nums []int) int {
	if len(nums) == 1 {
		return 1
	} else {
		var direction, ans int
		for i := 1; i < len(nums); i++ {
			if nums[i] > nums[i-1] {
				if direction <= 0 {
					direction = 1
					ans++
				}
			} else if nums[i] < nums[i-1] {
				if direction >= 0 {
					direction = -1
					ans++
				}
			}
		}
		return ans + 1
	}
}

//	func maxSubArray(nums []int) int {
//		sum := 0
//		maxSum := math.MinInt
//		for i := 0; i < len(nums); i++ {
//			j := i
//			for ; j < len(nums); j++ {
//				sum += nums[j]
//				maxSum = max(maxSum, sum)
//				if sum < 0 {
//					i = j + 1
//					sum = 0
//				}
//			}
//			if j == len(nums) {
//				break
//			}
//		}
//		return maxSum
//	}
//
//	func maxProfit(prices []int) int {
//		profit := 0
//		for i := 1; i < len(prices); i++ {
//			if prices[i] > prices[i-1] {
//				profit += prices[i] - prices[i-1]
//			}
//		}
//		return profit
//	}
//

func canJump(nums []int) bool {
	right := 0
	for i := 0; i <= right; i++ {
		right = max(right, nums[i]+i)
		if right >= len(nums)-1 {
			return true
		}
	}
	return false
}
func jump(nums []int) int {
	var maxRange, maxRangeIndex, pos, ans int
	if len(nums) == 1 {
		return 0
	}
	for {
		ans++
		if nums[pos]+pos >= len(nums)-1 {
			break
		} else {
			for i := pos + 1; i <= nums[pos]+pos; i++ {
				if nums[i]+i > maxRange {
					maxRange = nums[i] + i
					maxRangeIndex = i
				}
			}
			pos = maxRangeIndex
		}
	}
	return ans
}
func oddCells(m int, n int, indices [][]int) int {
	rowCount := make(map[int]int)
	colCount := make(map[int]int)
	for _, index := range indices {
		rowCount[index[0]]++
		colCount[index[1]]++
	}
	count := 0
	for _, v := range rowCount {
		if v%2 != 0 {
			count += n
		}
	}
	aver := count / n
	for _, v := range colCount {
		if v%2 != 0 {
			count = count - aver + m - aver
		}
	}
	return count
}

func reformatDate(date string) string {
	dates := strings.Split(date, " ")
	months := map[string]string{
		"Jan": "01",
		"Feb": "02",
		"Mar": "03",
		"Apr": "04",
		"May": "05",
		"Jun": "06",
		"Jul": "07",
		"Aug": "08",
		"Sep": "09",
		"Oct": "10",
		"Nov": "11",
		"Dec": "12",
	}
	year := dates[2]
	month := months[dates[1]]
	day := fmt.Sprintf("%02s", dates[0][:len(dates[0])-2])
	return year + "-" + month + "-" + day
}
func findLeastNumOfUniqueInts(arr []int, k int) int {
	count := make(map[int]int)
	for _, num := range arr {
		count[num]++
	}
	ans := len(count)
	n := 1
	for k > 0 {
		for _, v := range count {
			if v == n {
				if k >= n {
					k -= n
					ans--
				} else {
					k = 0
				}
			}
		}
		n++
	}
	return ans
}
func decode(encoded []int, first int) []int {
	ans := make([]int, len(encoded)+1)
	ans[0] = first
	for i, e := range encoded {
		ans[i+1] = ans[i] ^ e
	}
	return ans
}
func lemonadeChange(bills []int) bool {
	count := make([]int, 2)
	for _, bill := range bills {
		if bill == 5 {
			count[0]++
		} else if bill == 10 {
			if count[0] == 0 {
				return false
			} else {
				count[0]--
				count[1]++
			}
		} else if bill == 20 {
			if count[1] == 0 {
				if count[0] < 3 {
					return false
				} else {
					count[0] -= 3
				}
			} else if count[1] >= 1 {
				if count[0] == 0 {
					return false
				} else {
					count[0]--
					count[1]--
				}
			}
		}
	}
	return true
}

//
//type CustomStack struct {
//	stack   []int
//	maxSize int
//}
//
//func Constructor(maxSize int) CustomStack {
//	return CustomStack{stack: make([]int, 0, maxSize), maxSize: maxSize}
//}
//
//func (this *CustomStack) Push(x int) {
//	if len(this.stack) < this.maxSize {
//		this.stack = append(this.stack, x)
//	}
//}
//
//func (this *CustomStack) Pop() int {
//	if len(this.stack) != 0 {
//		val := this.stack[len(this.stack)-1]
//		this.stack = this.stack[:len(this.stack)-1]
//		return val
//	} else {
//		return -1
//	}
//}
//
//func (this *CustomStack) Increment(k int, val int) {
//	k = min(len(this.stack), k)
//	for i := 0; i < k; i++ {
//		this.stack[i] += val
//	}
//}

//	func equalPairs(grid [][]int) int {
//		count := 0
//		newGrid := transpose(grid)
//		for _, row1 := range grid {
//			for _, row2 := range newGrid {
//				if slices.Equal(row1, row2) {
//					count++
//				}
//			}
//		}
//		return count
//	}
func distributeCandies(candies int, num_people int) []int {
	ans := make([]int, num_people)
	index := 0
	candiesNum := 1
	for candies > 0 {
		if candies > candiesNum {
			ans[index%num_people] += candiesNum
			candies -= candiesNum
		} else {
			ans[index%num_people] += candies
			candies = 0
		}
		candiesNum++
		index++
	}
	return ans
}
func minSubsequence(nums []int) (ans []int) {
	slices.Sort(nums)
	sum := 0
	for _, num := range nums {
		sum += num
	}
	s := 0
	for i := len(nums) - 1; i >= 0; i-- {
		if s <= sum-s {
			s += nums[i]
			ans = append(ans, nums[i])
		}
	}
	return
}
func countConsistentStrings(allowed string, words []string) (count int) {
	for _, word := range words {
		flag := true
		for _, char := range word {
			if !strings.Contains(allowed, string(char)) {
				flag = false
				break
			}
		}
		if flag {
			count++
		}
	}
	return
}
func mostFrequent(nums []int, key int) int {
	hash := make(map[int]int)
	maxFreq := math.MinInt
	ans := 0
	for i := 0; i < len(nums)-1; i++ {
		if nums[i] == key {
			hash[nums[i+1]]++
			if hash[nums[i+1]] > maxFreq {
				maxFreq = hash[nums[i+1]]
				ans = nums[i+1]
			}
		}
	}
	return ans
}
func maxSum(grid [][]int) int {
	maximumSum := math.MinInt
	for i := 0; i <= len(grid)-3; i++ {
		for j := 0; j <= len(grid[0])-3; j++ {
			maximumSum = max(maximumSum, grid[i][j]+grid[i][j+1]+grid[i][j+2]+grid[i+1][j+1]+grid[i+2][j]+grid[i+2][j+1]+grid[i+2][j+2])
		}
	}
	return maximumSum
}

func sortSentence(s string) string {
	words := strings.Split(s, " ")
	slices.SortFunc(words, func(a, b string) int {
		return int(a[len(a)-1]) - int(b[len(b)-1])
	})
	builder := strings.Builder{}
	for i, word := range words {
		builder.WriteString(word[:len(word)-1])
		if i != len(words)-1 {
			builder.WriteByte(' ')
		}
	}
	return builder.String()
}
func maximumValue(strs []string) int {
	maxVal := 0
	for _, str := range strs {
		val, err := strconv.Atoi(str)
		if err != nil {
			val = len(str)
		}
		maxVal = max(maxVal, val)
	}
	return maxVal
}

//	func minOperations(n int) int {
//		count := 0
//		for i := 0; i < n/2; i++ {
//			count += (2*(n-i-1) + 1 - (2*i + 1)) / 2
//		}
//		return count
//	}
func sortTheStudents(score [][]int, k int) [][]int {
	slices.SortFunc(score, func(a, b []int) int {
		return b[k] - a[k]
	})
	return score
}
func totalMoney(n int) int {
	sum := 0
	for i := 0; i < n/7; i++ {
		sum += 28 + 7*i
	}
	for i := n/7 + 1; i <= n/7+n%7; i++ {
		sum += i
	}
	return sum
}

func removeAnagrams(words []string) (ans []string) {
	preCount := make(map[byte]int)
	for i, word := range words {
		count := make(map[byte]int)
		for _, char := range word {
			count[byte(char)]++
		}
		if i == 0 {
			preCount = count
			ans = append(ans, word)
		} else {
			if !reflect.DeepEqual(preCount, count) {
				ans = append(ans, word)
				preCount = count
			}
		}
	}
	return
}

//	func minOperations(boxes string) []int {
//		length := len(boxes)
//		haveBalls := make([]int, 0, length)
//		for i, c := range boxes {
//			if c == '1' {
//				haveBalls = append(haveBalls, i)
//			}
//		}
//		ans := make([]int, length)
//		for i := 0; i < length; i++ {
//			res := 0
//			for j := 0; j < len(haveBalls); j++ {
//				res += Abs(haveBalls[j] - i)
//			}
//			ans[i] = res
//		}
//		return ans
//	}
func findKthPositive(arr []int, k int) int {
	n := 1
	i := 0
	for ; i < len(arr) && k > 0; n++ {
		if arr[i] != n {
			k--
		} else {
			i++
		}
	}
	if i >= len(arr) {
		for ; k > 0; k-- {
			n++
		}
	}
	return n - 1
}
