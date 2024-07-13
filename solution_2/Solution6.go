package main

import (
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

func kWeakestRows(mat [][]int, k int) (ans []int) {
	senryoku := make([][2]int, len(mat))
	for i, row := range mat {
		senryoku[i][0] = i
		for _, val := range row {
			if val == 1 {
				senryoku[i][1]++
			}
		}
	}
	slices.SortFunc(senryoku, func(a, b [2]int) int {
		if a[1] != b[1] {
			return a[1] - b[1]
		} else {
			return a[0] - b[0]
		}
	})
	for i := 0; i < k; i++ {
		ans = append(ans, senryoku[i][0])
	}
	return
}
func checkIfExist(arr []int) bool {
	for i := 0; i < len(arr); i++ {
		for j := 0; j < len(arr); j++ {
			if i != j && arr[i] == 2*arr[j] {
				return true
			}
		}
	}
	return false
}
func minimumOperations(nums []int) (count int) {
	slices.Sort(nums)
	for i := 0; i < len(nums); i++ {
		if nums[i] > 0 {
			count++
			for j := i + 1; j < len(nums); j++ {
				nums[j] -= nums[i]
			}
			nums[i] = 0
		}
	}
	return
}
func backspaceCompare(s string, t string) bool {
	stackS := make([]byte, 0, len(s))
	stackT := make([]byte, 0, len(t))
	for _, char := range s {
		if char != '#' {
			stackS = append(stackS, byte(char))
		} else if char == '#' && len(stackS) != 0 {
			stackS = stackS[:len(stackS)-1]
		}
	}
	for _, char := range t {
		if char != '#' {
			stackT = append(stackT, byte(char))
		} else if char == '#' && len(stackT) != 0 {
			stackT = stackT[:len(stackT)-1]
		}
	}
	return string(stackS) == string(stackT)
}
func isFascinating(n int) bool {
	str := strconv.Itoa(n) + strconv.Itoa(2*n) + strconv.Itoa(3*n)
	onceMap := make(map[byte]struct{})
	for _, char := range str {
		if char == '0' {
			return false
		}
		if _, ok := onceMap[byte(char)]; ok {
			return false
		} else {
			onceMap[byte(char)] = struct{}{}
		}
	}
	return true
}
func sumOfUnique(nums []int) (sum int) {
	countMap := make(map[int]int)
	for _, num := range nums {
		countMap[num]++
	}
	for k, v := range countMap {
		if v == 1 {
			sum += k
		}
	}
	return
}
func reverseOnlyLetters(s string) string {
	left, right := 0, len(s)-1
	bytes := []byte(s)
	for {
		for left < len(bytes) && !unicode.IsLetter(rune(bytes[left])) {
			left++
		}
		for right >= 0 && !unicode.IsLetter(rune(bytes[right])) {
			right--
		}
		if left >= right {
			break
		}
		bytes[left], bytes[right] = bytes[right], bytes[left]
		left++
		right--
	}
	return string(bytes)
}
func countGoodRectangles(rectangles [][]int) int {
	maxLen := math.MinInt
	length := 0
	lenMap := make(map[int]int)
	for _, rectangle := range rectangles {
		length = min(rectangle[0], rectangle[1])
		lenMap[length]++
		maxLen = max(maxLen, length)
	}
	return lenMap[maxLen]
}
func numOfStrings(patterns []string, word string) (count int) {
	for _, pattern := range patterns {
		if strings.Contains(word, pattern) {
			count++
		}
	}
	return
}
func kthFactor(n int, k int) int {
	for i := 1; i <= n; i++ {
		if n%i == 0 {
			k--
			if k == 0 {
				return i
			}
		}
	}
	return -1
}
func isPrefixString(s string, words []string) bool {
	var t string
	for _, word := range words {
		t += word
		if !strings.HasPrefix(s, t) {
			return false
		}
		if t == s {
			return true
		}
	}
	return false
}
func distanceBetweenBusStops(distance []int, start int, destination int) int {
	var distance1, distance2 int
	for i := start; i != destination; {
		distance1 += distance[i]
		if i == len(distance)-1 {
			i = 0
		} else {
			i++
		}
	}
	for i := start; i != destination; {
		if i == 0 {
			i = len(distance) - 1
		} else {
			i--
		}
		distance2 += distance[i]
	}
	return min(distance1, distance2)
}
func bitwiseComplement(n int) int {
	if n == 0 {
		return 1
	}
	var digitNum int
	num := n
	for n > 0 {
		digitNum++
		n = n >> 1
	}
	return (1 << digitNum) - num - 1
}
func addToArrayForm(num []int, k int) []int {
	carry := 0
	for i := len(num) - 1; i >= 0; i-- {
		num[i] += carry + k%10
		carry = num[i] / 10
		num[i] %= 10
		k /= 10
	}
	for ; k > 0; k /= 10 {
		num = append([]int{(carry + k%10) % 10}, num...)
		carry = (carry + k%10) / 10
	}
	if carry != 0 {
		num = append([]int{1}, num...)
	}
	return num
}
func binarySearch(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			return mid
		}
	}
	return -1
}
func findFinalValue(nums []int, original int) int {
	slices.Sort(nums)
	index := binarySearch(nums, original)
	for index != -1 {
		nums = nums[index:]
		original *= 2
		index = binarySearch(nums, original)
	}
	return original
}
func rearrangeArray(nums []int) (ans []int) {
	posNums := make([]int, 0, len(nums)/2)
	negNums := make([]int, 0, len(nums)/2)
	for _, num := range nums {
		if num < 0 {
			negNums = append(negNums, num)
		} else {
			posNums = append(posNums, num)
		}
	}
	for i := 0; i < len(posNums); i++ {
		ans = append(ans, posNums[i], negNums[i])
	}
	return
}

//	func maxScore(s string) int {
//		leftScore, rightScore := 0, len(s)-strings.Count(s, "0")
//		var score int
//		for i := 0; i < len(s)-1; i++ {
//			if s[i] == '0' {
//				leftScore++
//			} else {
//				rightScore--
//			}
//			score = max(score, leftScore+rightScore)
//		}
//		return score
//	}
//
//	func maxDistance(colors []int) int {
//		maxDist := 0
//		for i := 0; i < len(colors); i++ {
//			for j := i + 1; j < len(colors); j++ {
//				if colors[i] != colors[j] {
//					maxDist = max(maxDist, j-i)
//				}
//			}
//		}
//		return maxDist
//	}
func minMaxGame(nums []int) int {
	for {
		if len(nums) == 1 {
			return nums[0]
		}
		minFlag := true
		newNums := make([]int, 0, len(nums)/2)
		for i := 0; i < len(nums); i += 2 {
			if minFlag {
				newNums = append(newNums, min(nums[i], nums[i+1]))
			} else {
				newNums = append(newNums, max(nums[i], nums[i+1]))
			}
			minFlag = !minFlag
		}
		nums = newNums
	}
}
func strongPasswordCheckerII(password string) bool {
	if len(password) < 8 {
		return false
	} else {
		var hasLowerLetter, hasUpperLetter, hasDigit, hasSpecial bool
		for i, char := range password {
			if i != 0 && byte(char) == password[i-1] {
				return false
			}
			if !hasLowerLetter && unicode.IsLower(char) {
				hasLowerLetter = true
			}
			if !hasUpperLetter && unicode.IsUpper(char) {
				hasUpperLetter = true
			}
			if !hasDigit && unicode.IsDigit(char) {
				hasDigit = true
			}
			if !hasSpecial && strings.Contains("!@#$%^&*()-+", string(char)) {
				hasSpecial = true
			}
		}
		if hasLowerLetter && hasUpperLetter && hasDigit && hasSpecial {
			return true
		}
		return false
	}
}
func bestHand(ranks []int, suits []byte) string {
	suitMap := make(map[byte]struct{})
	rankMap := make(map[int]int)
	for _, suit := range suits {
		if _, ok := suitMap[suit]; !ok {
			suitMap[suit] = struct{}{}
		}
	}
	if len(suitMap) == 1 {
		return "Flush"
	}
	maxRank := 0
	for _, rank := range ranks {
		rankMap[rank]++
		maxRank = max(maxRank, rankMap[rank])
	}
	if maxRank >= 3 {
		return "Three of a Kind"
	} else if maxRank >= 2 {
		return "Pair"
	}
	return "High Card"
}
func minNumber(nums1 []int, nums2 []int) int {
	slices.Sort(nums1)
	slices.Sort(nums2)
	hash := make(map[int]struct{})
	for _, num := range nums1 {
		if _, ok := hash[num]; !ok {
			hash[num] = struct{}{}
		}
	}
	num1 := math.MaxInt
	for _, num := range nums2 {
		if _, ok := hash[num]; ok {
			num1 = num
			break
		}
	}
	return min(num1, nums1[0]*10+nums2[0], nums1[0]+nums2[0]*10)
}

//	func reformat(s string) string {
//		digitArr := make([]byte, 0)
//		letterArr := make([]byte, 0)
//		for _, char := range s {
//			if unicode.IsLetter(char) {
//				letterArr = append(letterArr, byte(char))
//			} else if unicode.IsDigit(char) {
//				digitArr = append(digitArr, byte(char))
//			}
//		}
//		ans := make([]byte, 0)
//		if Abs(len(letterArr)-len(digitArr)) > 1 {
//			return string(ans)
//		} else {
//			length := min(len(letterArr), len(digitArr))
//			for i := 0; i < length; i++ {
//				ans = append(ans, letterArr[i], digitArr[i])
//			}
//			if len(letterArr) > len(digitArr) {
//				ans = append(ans, letterArr[length:]...)
//			} else {
//				ans = append(digitArr[length:], ans...)
//			}
//			return string(ans)
//		}
//	}
func minAddToMakeValid(s string) int {
	stack := make([]byte, 0)
	for _, char := range s {
		if char == '(' {
			stack = append(stack, '(')
		} else {
			if len(stack) == 0 || stack[len(stack)-1] != '(' {
				stack = append(stack, ')')
			} else {
				stack = stack[:len(stack)-1]
			}
		}
	}
	return len(stack)
}
func areOccurrencesEqual(s string) bool {
	countMap := make(map[byte]int)
	for _, char := range s {
		countMap[byte(char)]++
	}
	val := -1
	for _, v := range countMap {
		if val == -1 {
			val = v
		}
		if v != val {
			return false
		}
	}
	return true
}
func greatestLetter(s string) string {
	greatLetterArr := make([]byte, 26)
	for _, char := range s {
		if unicode.IsLower(char) {
			greatLetterArr[unicode.ToUpper(char)-'A'] |= 1
		} else if unicode.IsUpper(char) {
			greatLetterArr[char-'A'] |= 10
		}
	}
	for i := 25; i >= 0; i-- {
		if greatLetterArr[i] == 11 {
			return string(byte('A' + i))
		}
	}
	return ""
}
func minimizedStringLength(s string) int {
	hash := make(map[byte]struct{})
	for _, char := range s {
		if _, ok := hash[byte(char)]; !ok {
			hash[byte(char)] = struct{}{}
		}
	}
	return len(hash)
}
func flipAndInvertImage(image [][]int) [][]int {
	for _, row := range image {
		slices.Reverse(row)
		for i, num := range row {
			row[i] = 1 - num
		}
	}
	return image
}

//	func checkDistances(s string, distance []int) bool {
//		dist := make(map[byte][]int)
//		for i, char := range s {
//			dist[byte(char)] = append(dist[byte(char)], i)
//		}
//		for k, v := range dist {
//			if Abs(v[0]-v[1])-1 != distance[k-'a'] {
//				return false
//			}
//		}
//		return true
//	}
func numWaterBottles(numBottles int, numExchange int) (count int) {
	emptyBottles := 0
	for numBottles > 0 {
		count += numBottles
		emptyBottles += numBottles
		numBottles = emptyBottles / numExchange
		emptyBottles %= numExchange
	}
	return
}
func maximumDifference(nums []int) int {
	maxSub := -1
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[j]-nums[i] > 0 {
				maxSub = max(maxSub, nums[j]-nums[i])
			}
		}
	}
	return maxSub
}
func numberOfCuts(n int) int {
	if n == 1 {
		return 0
	} else if n%2 == 0 {
		return n / 2
	} else {
		return n
	}
}
func checkStraightLine(coordinates [][]int) bool {
	if len(coordinates) == 2 {
		return true
	}
	var k, b float64
	if coordinates[0][0]-coordinates[1][0] != 0 {
		k = float64(coordinates[0][1]-coordinates[1][1]) / float64(coordinates[0][0]-coordinates[1][0])
		b = float64(coordinates[0][1]) - k*float64(coordinates[0][0])
		for i := 2; i < len(coordinates); i++ {
			if k*float64(coordinates[i][0])+b != float64(coordinates[i][1]) {
				return false
			}
		}
	} else {
		for i := 2; i < len(coordinates); i++ {
			if coordinates[i][0] != coordinates[0][0] {
				return false
			}
		}
	}
	return true
}
func diffChar(b1 byte, b2 byte, b3 byte) bool {
	if b1 != b2 && b1 != b3 && b2 != b3 {
		return true
	}
	return false
}
func countGoodSubstrings(s string) (count int) {
	for i := 0; i < len(s)-2; i++ {
		if diffChar(s[i], s[i+1], s[i+2]) {
			count++
		}
	}
	return
}
func largestOddNumber(num string) string {
	for i := len(num) - 1; i >= 0; i-- {
		digit := num[i] - '0'
		if digit%2 != 0 {
			return num[:i+1]
		}
	}
	return ""
}
func maximumBags(capacity []int, rocks []int, additionalRocks int) int {
	spare := make([]int, len(capacity))
	for i := 0; i < len(capacity); i++ {
		spare[i] = capacity[i] - rocks[i]
	}
	slices.Sort(spare)
	sum := 0
	for i := 0; i < len(spare); i++ {
		sum += spare[i]
		if sum > additionalRocks {
			return i
		}
	}
	return len(spare)
}
func getCommon(nums1 []int, nums2 []int) int {
	for _, num := range nums1 {
		if binarySearch(nums2, num) != -1 {
			return num
		}
	}
	return -1
}
func findSubarrays(nums []int) bool {
	for i := 0; i < len(nums)-1; i++ {
		a, b := nums[i], nums[i+1]
		for j := i + 1; j < len(nums)-1; j++ {
			if nums[j]+nums[j+1] == a+b {
				return true
			}
		}
	}
	return false
}
func distinctAverages(nums []int) int {
	slices.Sort(nums)
	set := make(map[float64]struct{})
	for len(nums) > 0 {
		aver := float64(nums[0]+nums[len(nums)-1]) / 2
		if _, ok := set[aver]; !ok {
			set[aver] = struct{}{}
		}
		nums = nums[1 : len(nums)-1]
	}
	return len(set)
}
func countAsterisks(s string) (count int) {
	barPos := make([]int, 0)
	for i, char := range s {
		if char == '|' {
			barPos = append(barPos, i)
		}
	}
	start := 0
	for j := 0; j < len(barPos); j += 2 {
		for ; start < barPos[j]; start++ {
			if s[start] == '*' {
				count++
			}
		}
		start = barPos[j+1]
	}
	for ; start < len(s); start++ {
		if s[start] == '*' {
			count++
		}
	}
	return
}
func maxIceCream(costs []int, coins int) int {
	slices.Sort(costs)
	for i, cost := range costs {
		coins -= cost
		if coins < 0 {
			return i
		}
	}
	return len(costs)
}
func sortEvenOdd(nums []int) (ans []int) {
	odd := make([]int, 0, len(nums)/2)
	even := make([]int, 0, len(nums)/2)
	for i, num := range nums {
		if i%2 == 0 {
			even = append(even, num)
		} else {
			odd = append(odd, num)
		}
	}
	slices.Sort(odd)
	slices.Reverse(odd)
	slices.Sort(even)
	i := 0
	for ; i < len(odd); i++ {
		ans = append(ans, even[i], odd[i])
	}
	ans = append(ans, odd[i:]...)
	ans = append(ans, even[i:]...)
	return
}

//	func minSteps(s string, t string) (count int) {
//		countS := make(map[byte]int, 26)
//		countT := make(map[byte]int, 26)
//		for _, char := range s {
//			countS[byte(char)]++
//		}
//		for _, char := range t {
//			countT[byte(char)]++
//		}
//		for i := 'a'; i <= 'z'; i++ {
//			count += Abs(countT[byte(i)] - countS[byte(i)])
//		}
//		return
//	}
func cellsInRange(s string) (ans []string) {
	rowStart := int(s[1] - '0')
	rowMax := int(s[4] - '0')
	for i := s[0]; i <= s[3]; i++ {
		for j := rowStart; j <= rowMax; j++ {
			ans = append(ans, string(i)+strconv.Itoa(j))
		}
	}
	return
}
func digitCount(num string) bool {
	countMap := make(map[int]int)
	for _, digit := range num {
		countMap[int(digit-'0')]++
	}
	for i, digit := range num {
		if countMap[i] != int(digit-'0') {
			return false
		}
	}
	return true
}
func getLucky(s string, k int) int {
	builder := strings.Builder{}
	for _, char := range s {
		builder.WriteString(strconv.Itoa(int(char - 'a' + 1)))
	}
	temp := builder.String()
	ans := 0
	for k > 0 {
		ans = 0
		for _, char := range temp {
			ans += int(char - '0')
		}
		temp = strconv.Itoa(ans)
		k--
	}
	return ans
}
func isBoomerang(points [][]int) bool {
	if !slices.Equal(points[0], points[1]) && !slices.Equal(points[0], points[2]) && !slices.Equal(points[1], points[2]) {
		if points[0][0] == points[1][0] {
			return points[0][0] != points[2][0]
		} else {
			var k float64
			k = float64(points[1][1]-points[0][1]) / float64(points[1][0]-points[0][0])
			return k != float64(points[2][1]-points[0][1])/float64(points[2][0]-points[0][0])
		}
	}
	return false
}
func largestAltitude(gain []int) int {
	curAlti, maxAlti := 0, 0
	for _, val := range gain {
		curAlti += val
		maxAlti = max(maxAlti, curAlti)
	}
	return maxAlti
}
func digitSumEven(num int) bool {
	sum := 0
	for num > 0 {
		sum += num % 10
		num /= 10
	}
	return sum%2 == 0
}
func countEven(num int) (ans int) {
	for i := 2; i <= num; i++ {
		if digitSumEven(i) {
			ans++
		}
	}
	return
}
func sumOfThree(num int64) (ans []int64) {
	if 3*(num/3) == num {
		return []int64{num/3 - 1, num / 3, num/3 + 1}
	} else {
		return []int64{}
	}
}
func areNumbersAscending(s string) bool {
	tokens := strings.Split(s, " ")
	curNum := -1
	for _, token := range tokens {
		if num, err := strconv.Atoi(token); err == nil {
			if curNum == -1 {
				curNum = num
			} else {
				if curNum >= num {
					return false
				}
				curNum = num
			}
		}
	}
	return true
}
func mostWordsFound(sentences []string) int {
	maxWordNum := math.MinInt
	for _, sentence := range sentences {
		maxWordNum = max(maxWordNum, len(strings.Split(sentence, " ")))
	}
	return maxWordNum
}
func freqAlphabets(s string) string {
	ans := strings.Builder{}
	for i := 0; i < len(s); {
		if i+2 < len(s) && s[i+2] == '#' {
			ans.WriteByte((s[i]-'0')*10 + s[i+1] - '0' + 'a' - 1)
			i += 3
		} else {
			ans.WriteByte(s[i] - '0' + 'a' - 1)
			i += 1
		}
	}
	return ans.String()
}
func countOne(num int) (count int) {
	for ; num > 0; num = num >> 1 {
		if num&1 == 1 {
			count++
		}
	}
	return
}
func sortByBits(arr []int) []int {
	slices.SortFunc(arr, func(a, b int) int {
		if countOne(a)-countOne(b) != 0 {
			return countOne(a) - countOne(b)
		} else {
			return a - b
		}
	})
	return arr
}
func transpose(matrix [][]int) [][]int {
	width, height := len(matrix), len(matrix[0])
	transposed := make([][]int, 0)
	for i := 0; i < height; i++ {
		transposed = append(transposed, make([]int, width))
	}
	for i, row := range matrix {
		for j, num := range row {
			transposed[j][i] = num
		}
	}
	return transposed
}
func mostFrequentEven(nums []int) int {
	count := make(map[int]int)
	maxCount := -1
	maxNum := -1
	for _, num := range nums {
		if num%2 == 0 {
			count[num]++
			if count[num] > maxCount {
				maxCount = count[num]
				maxNum = num
			} else if count[num] == maxCount {
				maxNum = min(maxNum, num)
			}
		}
	}
	return maxNum
}
func uncommonFromSentences(s1 string, s2 string) (ans []string) {
	count1 := make(map[string]int)
	count2 := make(map[string]int)
	words1 := strings.Split(s1, " ")
	words2 := strings.Split(s2, " ")
	for _, word := range words1 {
		count1[word]++
	}
	for _, word := range words2 {
		count2[word]++
	}
	for k, v := range count1 {
		if v == 1 && !slices.Contains(words2, k) {
			ans = append(ans, k)
		}
	}
	for k, v := range count2 {
		if v == 1 && !slices.Contains(words1, k) {
			ans = append(ans, k)
		}
	}
	return
}

//	func nearestValidPoint(x int, y int, points [][]int) int {
//		nearestIndex := -1
//		nearestDist := math.MaxInt
//		for i, point := range points {
//			if point[0] == x || point[1] == y {
//				if Abs(point[0]-x)+Abs(point[1]-y) < nearestDist {
//					nearestDist = Abs(point[0]-x) + Abs(point[1]-y)
//					nearestIndex = i
//				}
//			}
//		}
//		return nearestIndex
//	}
func findTheArrayConcVal(nums []int) int64 {
	var concVal int64
	for len(nums) > 0 {
		if len(nums) > 1 {
			val, _ := strconv.Atoi(strconv.Itoa(nums[0]) + strconv.Itoa(nums[len(nums)-1]))
			concVal += int64(val)
			nums = nums[1 : len(nums)-1]
		} else {
			concVal += int64(nums[0])
			nums = []int{}
		}
	}
	return concVal
}
func countDigits(num int) (count int) {
	temp := num
	for temp > 0 {
		if num%(temp%10) == 0 {
			count++
		}
		temp /= 10
	}
	return
}
func lowerBound(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return left
}

func searchRange(nums []int, target int) []int {
	start := lowerBound(nums, target)
	if start == len(nums) || nums[start] != target {
		return []int{-1, -1}
	}
	end := lowerBound(nums, target+1) - 1
	return []int{start, end}
}
func minimumCost(cost []int) (ans int) {
	slices.Sort(cost)
	for len(cost) >= 3 {
		ans += cost[len(cost)-1] + cost[len(cost)-2]
		cost = cost[:len(cost)-3]
	}
	for i := 0; i < len(cost); i++ {
		ans += cost[i]
	}
	return
}
func duplicateZeros(arr []int) {
	length := len(arr)
	for i := 0; i < length; i++ {
		if arr[i] == 0 {
			for j := length - 1; j > i; j-- {
				arr[j] = arr[j-1]
			}
			i++
		}
	}
}
func isCircularSentence(sentence string) bool {
	words := strings.Split(sentence, " ")
	for i := 0; i < len(words); i++ {
		if i > 0 && words[i][0] != words[i-1][len(words[i-1])-1] {
			return false
		}
		if i == len(words)-1 && words[0][0] != words[i][len(words[i])-1] {
			return false
		}
	}
	return true
}
func maxVowels(s string, k int) int {
	var vowelCount int
	for i := 0; i < k; i++ {
		if strings.Contains("aeiou", string(s[i])) {
			vowelCount++
		}
	}
	maxVowelCount := vowelCount
	for i := k; i < len(s); i++ {
		if strings.Contains("aeiou", string(s[i])) {
			vowelCount++
		}
		if strings.Contains("aeiou", string(s[i-k])) {
			vowelCount--
		}
		maxVowelCount = max(maxVowelCount, vowelCount)
	}
	return maxVowelCount
}
func checkValid(matrix [][]int) bool {
	hashCol := make([]map[int]struct{}, len(matrix))
	for i := 0; i < len(hashCol); i++ {
		hashCol[i] = make(map[int]struct{})
	}
	for _, row := range matrix {
		hashRow := make(map[int]struct{})
		for j, num := range row {
			if _, ok := hashRow[num]; !ok {
				hashRow[num] = struct{}{}
			} else {
				return false
			}
			if _, ok := hashCol[j][num]; !ok {
				hashCol[j][num] = struct{}{}
			} else {
				return false
			}
		}
	}
	return true
}
