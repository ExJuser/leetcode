package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
)

//	type StockSpanner struct {
//		stack [][2]int
//		cur   int
//	}
//
//	func Constructor() StockSpanner {
//		return StockSpanner{stack: make([][2]int, 0), cur: 0}
//	}
//
//	func (this *StockSpanner) Next(price int) int {
//		if this.cur == 0 {
//			this.stack = append(this.stack, [2]int{price, 1})
//			this.cur++
//			return this.stack[this.cur-1][1]
//		} else {
//			if price < this.stack[this.cur-1][0] {
//				this.stack = append(this.stack, [2]int{price, 1})
//				this.cur++
//				return this.stack[this.cur-1][1]
//			} else {
//				p := this.cur - 1
//				res := 1
//				for p >= 0 && price >= this.stack[p][0] {
//					res += this.stack[p][1]
//					p = p - this.stack[p][1]
//				}
//				this.stack = append(this.stack, [2]int{price, res})
//				this.cur++
//				return this.stack[this.cur-1][1]
//			}
//		}
//	}
func findUnsortedSubarray(nums []int) int {
	newNums := make([]int, len(nums))
	copy(newNums, nums)
	slices.Sort(newNums)
	left, right := 0, len(nums)-1
	for left < len(nums) && newNums[left] == nums[left] {
		left++
	}
	for right >= 0 && newNums[right] == nums[right] {
		right--
	}
	return max(right-left+1, 0)
}
func mergeAlternately(word1 string, word2 string) string {
	builder := strings.Builder{}
	length := min(len(word1), len(word2))
	for i := 0; i < length; i++ {
		builder.Write([]byte{word1[i], word2[i]})
	}
	builder.WriteString(word1[length:])
	builder.WriteString(word2[length:])
	return builder.String()
}
func findTheDifference(s string, t string) byte {
	hashS := make(map[byte]int)
	hashT := make(map[byte]int)
	for _, char := range s {
		hashS[byte(char)] += 1
	}
	for _, char := range t {
		hashT[byte(char)] += 1
	}
	for k, v := range hashT {
		if hashS[k] != v {
			return k
		}
	}
	return ' '
}
func arraySign(nums []int) int {
	minus := 0
	for _, num := range nums {
		if num == 0 {
			return 0
		} else if num < 0 {
			minus++
		}
	}
	if minus%2 != 0 {
		return -1
	} else {
		return 1
	}
}
func isMonotonic(nums []int) bool {
	sortedNums := make([]int, len(nums))
	copy(sortedNums, nums)
	slices.Sort(sortedNums)
	if slices.Equal(sortedNums, nums) {
		return true
	} else {
		slices.Reverse(sortedNums)
		return slices.Equal(sortedNums, nums)
	}
}
func toLowerCase(s string) string {
	return strings.ToLower(s)
}
func validateStackSequences(pushed []int, popped []int) bool {
	stack := make([]int, 0, len(pushed))
	for _, push := range pushed {
		if push != popped[0] {
			for len(stack) > 0 && stack[len(stack)-1] == popped[0] {
				stack = stack[:len(stack)-1]
				popped = popped[1:]
			}
			stack = append(stack, push)
		} else {
			popped = popped[1:]
			continue
		}
	}
	for len(stack) > 0 && stack[len(stack)-1] == popped[0] {
		stack = stack[:len(stack)-1]
		popped = popped[1:]
	}
	return len(stack) == 0
}
func repeatedCharacter(s string) byte {
	exist := make(map[byte]struct{})
	for _, char := range s {
		if _, ok := exist[byte(char)]; ok {
			return byte(char)
		} else {
			exist[byte(char)] = struct{}{}
		}
	}
	return ' '
}

//func buildArray(nums []int) []int {
//	ans := make([]int, len(nums))
//	for i, num := range nums {
//		ans[i] = nums[num]
//	}
//	return ans
//}

func numIdenticalPairs(nums []int) (count int) {
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[i] == nums[j] {
				count++
			}
		}
	}
	return
}
func repeatedNTimes(nums []int) int {
	countMap := make(map[int]int)
	for i, num := range nums {
		countMap[num]++
		if i >= len(nums)/2 {
			if countMap[num] == len(nums)/2 {
				return num
			}
		}
	}
	return -1
}
func percentageLetter(s string, letter byte) int {
	return int(100 * float64(strings.Count(s, string(letter))) / float64(len(s)))
}
func numberOfSteps(num int) (count int) {
	for num > 0 {
		if num%2 == 0 {
			num /= 2
		} else {
			num -= 1
		}
		count++
	}
	return
}
func generateTheString(n int) string {
	builder := strings.Builder{}
	if n%2 == 0 {
		for i := 0; i < n-1; i++ {
			builder.WriteString("a")
		}
		builder.WriteString("b")
	} else {
		for i := 0; i < n; i++ {
			builder.WriteString("a")
		}
	}
	return builder.String()
}
func removeTrailingZeros(num string) string {
	return strings.TrimRight(num, "0")
}

func maxPower(s string) int {
	length, maxLength := 1, 1
	curChar := ' '
	for _, char := range s {
		if char == curChar {
			length++
		} else {
			maxLength = max(maxLength, length)
			curChar = char
			length = 1
		}
	}
	return max(maxLength, length)
}
func checkIfPangram(sentence string) bool {
	if len(sentence) < 26 {
		return false
	} else {
		exist := make(map[byte]struct{})
		for _, char := range sentence {
			if _, ok := exist[byte(char)]; !ok {
				exist[byte(char)] = struct{}{}
			}
			if len(exist) == 26 {
				return true
			}
		}
		return len(exist) == 26
	}
}
func prefixCount(words []string, pref string) (count int) {
	for _, word := range words {
		if strings.HasPrefix(word, pref) {
			count++
		}
	}
	return
}

func sumZero(n int) (ans []int) {
	if n == 1 {
		return []int{0}
	} else if n == 2 {
		return []int{1, -1}
	} else {
		for i := 0; i < n-1; i++ {
			ans = append(ans, i)
		}
		ans = append(ans, -(n-2)*(n-1)/2)
		return
	}
}

//	func findMaxK(nums []int) int {
//		maxK := -1
//		hash := make(map[int]struct{})
//		for _, num := range nums {
//			hash[num] = struct{}{}
//			if _, ok := hash[-num]; ok {
//				maxK = max(maxK, Abs(num))
//			}
//		}
//		return maxK
//	}
func smallestEqual(nums []int) int {
	for i, num := range nums {
		if i%10 == num {
			return i
		}
	}
	return -1
}
func findDelayedArrivalTime(arrivalTime int, delayedTime int) int {
	return (arrivalTime + delayedTime) % 24
}
func commonFactors(a int, b int) (count int) {
	for i := 1; i <= min(a, b); i++ {
		if a%i == 0 && b%i == 0 {
			count++
		}
	}
	return
}
func countOperations(num1 int, num2 int) (count int) {
	for num1 != 0 && num2 != 0 {
		if num1 >= num2 {
			num1 -= num2
		} else {
			num2 -= num1
		}
		count++
	}
	return
}

//	func reversePrefix(word string, ch byte) string {
//		index := strings.Index(word, string(ch))
//		if index != -1 {
//			return reverseString(word[:index+1]) + word[index+1:]
//		} else {
//			return word
//		}
//	}
func dayOfYear(date string) int {
	t, _ := time.Parse(time.DateOnly, date)
	return t.YearDay()
}
func numUniqueEmails(emails []string) int {
	emailSet := make(map[string]struct{})
	for _, email := range emails {
		e := strings.Split(email, "@")
		local, domain := e[0], e[1]
		local = strings.ReplaceAll(local, ".", "")
		plusIndex := strings.Index(local, "+")
		if plusIndex != -1 {
			local = local[:plusIndex]
		}
		emailSet[local+"@"+domain] = struct{}{}
	}
	fmt.Println(emailSet)
	return len(emailSet)
}
func countSeniors(details []string) (count int) {
	for _, detail := range details {
		age, _ := strconv.Atoi(detail[11:13])
		if age > 60 {
			count++
		}
	}
	return
}

func nextBeautifulNumber(n int) int {
	for i := n + 1; ; i++ {
		countMap := make(map[int]int)
		x := i
		for x > 0 {
			countMap[x%10] += 1
			x /= 10
		}
		flag := true
		for k, v := range countMap {
			if k != v {
				flag = false
				break
			}
		}
		if flag {
			return i
		}
	}
}
func countTestedDevices(batteryPercentages []int) (count int) {
	for i, battery := range batteryPercentages {
		if battery > 0 {
			count++
			for j := i + 1; j < len(batteryPercentages); j++ {
				batteryPercentages[j] = max(0, batteryPercentages[j]-1)
			}
		}
	}
	return count
}
func climbStairs(n int) int {
	ans := []int{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903}
	return ans[n-1]
}
func kLengthApart(nums []int, k int) bool {
	lastKIndex := -1
	for i, num := range nums {
		if num == 1 {
			if lastKIndex != -1 && i-lastKIndex-1 < k {
				return false
			} else {
				lastKIndex = i
			}
		}
	}
	return true
}
func maximum69Number(num int) int {
	numStr := strconv.Itoa(num)
	ans, _ := strconv.Atoi(strings.Replace(numStr, "6", "9", 1))
	return ans
}

//	func maximum69Number(num int) int {
//		numBytes := []byte(strconv.Itoa(num))
//		for i, char := range numBytes {
//			if char == '6' {
//				numBytes[i] = '9'
//				break
//			}
//		}
//		ans, _ := strconv.Atoi(string(numBytes))
//		return ans
//	}
func sortPeople(names []string, heights []int) (ans []string) {
	nameHeightMap := make(map[int]string)
	for i, height := range heights {
		nameHeightMap[height] = names[i]
	}
	slices.SortFunc(heights, func(a, b int) int {
		return b - a
	})
	for _, height := range heights {
		name := nameHeightMap[height]
		ans = append(ans, name)
	}
	return
}
func uniqueOccurrences(arr []int) bool {
	countMap := make(map[int]int)
	countMapMap := make(map[int]struct{})
	for _, num := range arr {
		countMap[num] += 1
	}
	for _, v := range countMap {
		if _, ok := countMapMap[v]; ok {
			return false
		} else {
			countMapMap[v] = struct{}{}
		}
	}
	return true
}
func lastStoneWeight(stones []int) int {
	for len(stones) > 1 {
		slices.Sort(stones)
		x := stones[len(stones)-2]
		y := stones[len(stones)-1]
		stones = stones[:len(stones)-2]
		if x < y {
			stones = append(stones, y-x)
		}
	}
	if len(stones) == 0 {
		return 0
	} else {
		return stones[0]
	}
}

//	func sortArrayByParityII(nums []int) []int {
//		oddNums := make([]int, 0, len(nums)/2)
//		evenNums := make([]int, 0, len(nums)/2)
//		for _, num := range nums {
//			if num%2 == 0 {
//				evenNums = append(evenNums, num)
//			} else {
//				oddNums = append(oddNums, num)
//			}
//		}
//		for i := 0; i < len(nums)/2; i++ {
//			nums[2*i] = evenNums[i]
//			nums[2*i+1] = oddNums[i]
//		}
//		return nums
//	}
func rowAndMaximumOnes(mat [][]int) []int {
	var maxOne, maxOneRow int
	for i, row := range mat {
		one := 0
		for _, num := range row {
			if num == 1 {
				one++
			}
		}
		if one > maxOne {
			maxOne = one
			maxOneRow = i
		}
	}
	return []int{maxOneRow, maxOne}
}
func countMatches(items [][]string, ruleKey string, ruleValue string) (count int) {
	var valueIndex int
	if ruleKey == "type" {
		valueIndex = 0
	} else if ruleKey == "color" {
		valueIndex = 1
	} else {
		valueIndex = 2
	}
	for _, item := range items {
		if item[valueIndex] == ruleValue {
			count++
		}
	}
	return
}
func sortArrayByParity(nums []int) []int {
	slices.SortStableFunc(nums, func(a, b int) int {
		return a%2 - b%2
	})
	return nums
}
func makeSmallestPalindrome(s string) string {
	bytes := []byte(s)
	for i := 0; i < len(bytes)/2; i++ {
		if bytes[i] != bytes[len(bytes)-i-1] {
			bytes[i] = min(bytes[i], bytes[len(bytes)-i-1])
			bytes[len(bytes)-i-1] = bytes[i]
		}
	}
	return string(bytes)
}

//	func minimumAbsDifference(arr []int) [][]int {
//		slices.Sort(arr)
//		absDiffMap := make(map[int][][]int)
//		minAbsDiff := Abs(arr[0] - arr[1])
//		for i := 0; i < len(arr)-1; i++ {
//			absDiff := Abs(arr[i] - arr[i+1])
//			minAbsDiff = min(minAbsDiff, absDiff)
//			absDiffMap[absDiff] = append(absDiffMap[absDiff], []int{arr[i], arr[i+1]})
//		}
//		return absDiffMap[minAbsDiff]
//	}
func sumEvenAfterQueries(nums []int, queries [][]int) (ans []int) {
	isEven := make([]bool, len(nums))
	sum := 0
	for i, num := range nums {
		if num%2 == 0 {
			sum += num
			isEven[i] = true
		}
	}
	for _, query := range queries {
		val, index := query[0], query[1]
		if isEven[index] && val%2 == 0 {
			sum += val
		} else if isEven[index] && val%2 != 0 {
			sum -= nums[index]
			isEven[index] = false
		} else if !isEven[index] && val%2 != 0 {
			sum += val + nums[index]
			isEven[index] = true
		}
		nums[index] += val
		ans = append(ans, sum)
	}
	return
}
func maximumCount(nums []int) int {
	minusIndex := sort.Search(len(nums), func(i int) bool {
		return nums[i] >= 0
	}) - 1
	plusIndex := minusIndex + 1
	for ; plusIndex < len(nums); plusIndex++ {
		if nums[plusIndex] > 0 {
			break
		}
	}
	return max(minusIndex+1, len(nums)-plusIndex)
}
func destCity(paths [][]string) string {
	tripMap := make(map[string]string)
	for _, path := range paths {
		tripMap[path[0]] = path[1]
	}
	for _, v := range tripMap {
		if _, ok := tripMap[v]; !ok {
			return v
		}
	}
	return ""
}
func theMaximumAchievableX(num int, t int) int {
	return num + t*2
}
func relativeSortArray(arr1 []int, arr2 []int) []int {
	indexMap := make(map[int]int)
	for i, num := range arr2 {
		indexMap[num] = i
	}
	slices.SortFunc(arr1, func(a, b int) int {
		x, oka := indexMap[a]
		y, okb := indexMap[b]
		if oka && okb {
			return x - y
		} else if !oka && !okb {
			return a - b
		} else if !oka {
			return 1
		} else {
			return -1
		}
	})
	return arr1
}

func word2Number(word string) int {
	bytesW := []byte(word)
	for i := 0; i < len(bytesW); i++ {
		bytesW[i] -= 'a' - '0'
	}
	ans, _ := strconv.Atoi(string(bytesW))
	return ans
}

func isSumEqual(firstWord string, secondWord string, targetWord string) bool {
	fmt.Println(word2Number(firstWord), word2Number(secondWord), word2Number(targetWord))
	return word2Number(firstWord)+word2Number(secondWord) == word2Number(targetWord)
}
func isSameAfterReversals(num int) bool {
	numBytes := []byte(strconv.Itoa(num))
	slices.Reverse(numBytes)
	newNum, _ := strconv.Atoi(string(numBytes))
	numBytes = []byte(strconv.Itoa(newNum))
	slices.Reverse(numBytes)
	newNum, _ = strconv.Atoi(string(numBytes))
	return newNum == num
}
func numberOfPairs(nums []int) []int {
	countMap := make(map[int]int)
	var pairCount, leftCount int
	for _, num := range nums {
		countMap[num] += 1
	}
	for _, v := range countMap {
		pairCount += v / 2
		leftCount += v % 2
	}
	return []int{pairCount, leftCount}
}

func findGCD(nums []int) int {
	minNum, maxNum := slices.Min(nums), slices.Max(nums)
	for maxNum%minNum != 0 {
		maxNum, minNum = minNum, maxNum%minNum
	}
	return minNum
}
func findSpecialInteger(arr []int) int {
	countMap := make(map[int]int)
	for _, num := range arr {
		countMap[num] += 1
		if 4*countMap[num] > len(arr) {
			return num
		}
	}
	return -1
}
func xorOperation(n int, start int) int {
	var ans int
	for i := 0; i < n; i++ {
		if i != 0 {
			ans ^= start + 2*i
		} else {
			ans = start + 2*i
		}
	}
	return ans
}
func sumOfMultiples(n int) (ans int) {
	for i := 1; i <= n; i++ {
		if i%3 == 0 || i%5 == 0 || i%7 == 0 {
			ans += i
		}
	}
	return
}
func maximumWealth(accounts [][]int) int {
	var maxWealth, wealth int
	for _, account := range accounts {
		wealth = 0
		for _, w := range account {
			wealth += w
		}
		maxWealth = max(maxWealth, wealth)
	}
	return maxWealth
}
func alternateDigitSum(n int) (ans int) {
	str := strconv.Itoa(n)
	sign := 1
	for _, char := range str {
		ans += sign * int(char-'0')
		sign *= -1
	}
	return ans
}
func checkOnesSegment(s string) bool {
	return !strings.Contains(s, "01")
}
func evenOddBit(n int) []int {
	var even, odd int
	for i := 9; i >= 0; i-- {
		if 1<<i <= n {
			if i%2 == 0 {
				even++
			} else {
				odd++
			}
			n -= 1 << i
		}
	}
	return []int{even, odd}
}
func pivotInteger(n int) int {
	leftSum := 0
	for i := 1; i <= n; i++ {
		leftSum += i
	}
	rightSum := n
	for i := n; i >= 1; i-- {
		if leftSum == rightSum {
			return i
		} else {
			leftSum -= i
			rightSum += i - 1
		}

	}
	return -1
}
func halvesAreAlike(s string) bool {
	left, right := strings.ToLower(s[:len(s)/2]), strings.ToLower(s[len(s)/2:])
	var vowelLeft, vowelRight int
	for i := 0; i < len(left); i++ {
		if strings.Contains("aeiou", string(left[i])) {
			vowelLeft++
		}
		if strings.Contains("aeiou", string(right[i])) {
			vowelRight++
		}
	}
	return vowelLeft == vowelRight
}
func findDifference(nums1 []int, nums2 []int) [][]int {
	numsMap1 := make(map[int]struct{})
	numsMap2 := make(map[int]struct{})
	for _, num := range nums1 {
		if _, ok := numsMap1[num]; !ok {
			numsMap1[num] = struct{}{}
		}
	}
	for _, num := range nums2 {
		if _, ok := numsMap2[num]; !ok {
			numsMap2[num] = struct{}{}
		}
	}
	var notInNums1, notInNums2 []int
	for k, _ := range numsMap2 {
		if _, ok := numsMap1[k]; !ok {
			notInNums1 = append(notInNums1, k)
		}
	}
	for k, _ := range numsMap1 {
		if _, ok := numsMap2[k]; !ok {
			notInNums2 = append(notInNums2, k)
		}
	}
	return [][]int{notInNums2, notInNums1}
}
func buyChoco(prices []int, money int) int {
	slices.Sort(prices)
	if prices[0]+prices[1] > money {
		return money
	} else {
		return money - prices[0] - prices[1]
	}
}
func luckyNumbers(matrix [][]int) (ans []int) {
	rowMin := make([]int, len(matrix))
	colMax := make([]int, len(matrix[0]))
	for i := 0; i < len(rowMin); i++ {
		rowMin[i] = math.MaxInt
	}
	for i := 0; i < len(rowMin); i++ {
		for j := 0; j < len(colMax); j++ {
			if matrix[i][j] < rowMin[i] {
				rowMin[i] = matrix[i][j]
			}
			if matrix[i][j] > colMax[j] {
				colMax[j] = matrix[i][j]
			}
		}
	}
	for _, num := range rowMin {
		if slices.Contains(colMax, num) {
			ans = append(ans, num)
		}
	}
	return
}
func validMountainArray(arr []int) bool {
	var i int
	for i+1 < len(arr) && arr[i] < arr[i+1] {
		i++
	}
	if i == 0 || i == len(arr)-1 {
		return false
	}
	for i+1 < len(arr) && arr[i] > arr[i+1] {
		i++
	}
	if i != len(arr)-1 {
		return false
	} else {
		return true
	}
}
func createTargetArray(nums []int, index []int) []int {
	ans := make([]int, 0, len(nums))
	for i := 0; i < len(nums); i++ {
		ans = append(ans[:index[i]], append([]int{nums[i]}, ans[index[i]:]...)...)
	}
	return ans
}

func countOdds(low int, high int) int {
	ans := (high - low) / 2
	if high%2 != 0 || low%2 != 0 {
		ans += 1
	}
	return ans
}
func minStartValue(nums []int) int {
	minSum := math.MaxInt
	sum := 0
	for _, num := range nums {
		sum += num
		minSum = min(minSum, sum)
	}
	return max(1, 1-minSum)
}
func kClosest(points [][]int, k int) (ans [][]int) {
	distances := make([][2]int, 0, len(points))
	for i, point := range points {
		distances = append(distances, [2]int{point[0]*point[0] + point[1]*point[1], i})
	}
	slices.SortFunc(distances, func(a, b [2]int) int {
		return a[0] - b[0]
	})
	for i := 0; i < k; i++ {
		ans = append(ans, points[distances[i][1]])
	}
	return
}

//func firstPalindrome(words []string) string {
//	for _, word := range words {
//		if isPalindrome(word) {
//			return word
//		}
//	}
//	return ""
//}

//	func countPairs(nums []int, k int) (count int) {
//		for i := 0; i < len(nums); i++ {
//			for j := i + 1; j < len(nums); j++ {
//				if nums[i] == nums[j] && (i*j)%k == 0 {
//					count++
//				}
//			}
//		}
//		return
//	}
func separateDigits(nums []int) (ans []int) {
	for i := len(nums) - 1; i >= 0; i-- {
		num := nums[i]
		for num > 0 {
			ans = append(ans, num%10)
			num /= 10
		}
	}
	slices.Reverse(ans)
	return
}
func getMinDistance(nums []int, target int, start int) int {
	minVal := math.MaxInt
	for i, num := range nums {
		if num == target {
			minVal = min(minVal, Abs(i-start))
		}
	}
	return minVal
}
func arrayStringsAreEqual(word1 []string, word2 []string) bool {
	builder1 := strings.Builder{}
	builder2 := strings.Builder{}
	for _, str := range word1 {
		builder1.WriteString(str)
	}
	for _, str := range word2 {
		builder2.WriteString(str)
	}
	return builder1.String() == builder2.String()
}
func countDistinctIntegers(nums []int) int {
	set := make(map[int]struct{}, len(nums)*2)
	for _, num := range nums {
		if _, ok := set[num]; !ok {
			set[num] = struct{}{}
		}
		if _, ok := set[reverseNum(num)]; !ok {
			set[reverseNum(num)] = struct{}{}
		}
	}
	return len(set)
}
func replaceElements(arr []int) []int {
	rightMax := make([]int, len(arr))
	maxVal := math.MinInt
	for i := len(arr) - 1; i >= 0; i-- {
		if i == len(arr)-1 {
			rightMax[i] = -1
		} else {
			rightMax[i] = maxVal
		}
		maxVal = max(maxVal, arr[i])
	}
	return rightMax
}
func balancedStringSplit(s string) (ans int) {
	count := make(map[byte]int)
	for _, char := range s {
		count[byte(char)] += 1
		if count['R'] == count['L'] {
			clear(count)
			ans++
		}
	}
	return
}
func threeConsecutiveOdds(arr []int) bool {
	for i := 0; i < len(arr)-2; i++ {
		if arr[i]%2 != 0 && arr[i+1]%2 != 0 && arr[i+2]%2 != 0 {
			return true
		}
	}
	return false
}
func longestContinuousSubstring(s string) int {
	maxLen := 0
	for i := 0; i < len(s); {
		j := i
		for j+1 < len(s) && s[j+1]-s[j] == 1 {
			j++
		}
		maxLen = max(maxLen, j-i+1)
		i = j + 1
	}
	return maxLen
}
func differenceOfSum(nums []int) int {
	var sum, digitsSum int
	for _, num := range nums {
		sum += num
		for num > 0 {
			digitsSum += num % 10
			num /= 10
		}
	}
	return Abs(sum - digitsSum)
}
func divideArray(nums []int) bool {
	countMap := make(map[int]int)
	for _, num := range nums {
		countMap[num] += 1
	}
	for _, v := range countMap {
		if v%2 != 0 {
			return false
		}
	}
	return true
}
func applyOperations(nums []int) []int {
	for i := 0; i < len(nums)-1; i++ {
		if nums[i] == nums[i+1] {
			nums[i] *= 2
			nums[i+1] = 0
		}
	}
	slices.SortStableFunc(nums, func(a, b int) int {
		if a != 0 && b != 0 {
			return 0
		} else if a == 0 {
			return 1
		} else {
			return -1
		}
	})
	return nums
}
