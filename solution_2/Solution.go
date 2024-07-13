package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
)

//	func getNext(substring []byte) (next []int) {
//		if len(substring) == 1 {
//			next = append(next, 0)
//			return
//		} else {
//			next = append(next, []int{0, 0}...)
//			cn := 0
//			for i := 2; i < len(substring); {
//				if substring[i-1] == substring[cn] {
//					next = append(next, cn+1)
//					cn++
//					i++
//				} else if cn > 0 {
//					cn = next[cn]
//				} else {
//					next = append(next, 0)
//					i++
//				}
//			}
//			return
//		}
//	}
func getNext(substring string) []int {
	if len(substring) == 1 {
		return []int{0}
	} else {
		next := make([]int, 0, len(substring))
		next = append(next, []int{0, 0}...)
		cn, i := 0, 2
		for i < len(substring) {
			if substring[i-1] == substring[cn] {
				next = append(next, cn+1)
				cn, i = cn+1, i+1
			} else if cn > 0 {
				cn = next[cn]
			} else {
				next = append(next, 0)
				i++
			}
		}
		return next
	}
}
func indexOf(mainString string, subString string) int {
	if len(mainString) < len(subString) {
		return -1
	} else {
		next := getNext(subString)
		i, j := 0, 0
		for i < len(mainString) && j < len(subString) {
			if mainString[i] == subString[j] {
				i, j = i+1, j+1
			} else if j > 0 {
				j = next[j]
			} else {
				i++
			}
		}
		if j == len(subString) {
			return i - j
		} else {
			return -1
		}
	}
}

//	func strStr(haystack string, needle string) int {
//		mainString := []byte(haystack)
//		subString := []byte(needle)
//		if len(subString) > len(mainString) {
//			return -1
//		} else {
//			i, j := 0, 0
//			next := getNext(subString)
//			for i < len(mainString) && j < len(subString) {
//				if mainString[i] == subString[j] {
//					i, j = i+1, j+1
//				} else if j > 0 {
//					j = next[j]
//				} else {
//					i++
//				}
//			}
//			if j == len(subString) {
//				return i - j
//			} else {
//				return -1
//			}
//		}
//	}
//
//	func indexOf(mainString string, subString string) int {
//		//mainString := []byte(a)
//		//subString := []byte(b)
//		if len(subString) > len(mainString) {
//			return -1
//		} else {
//			next := getNext(subString)
//			i, j := 0, 0
//			for i < len(mainString) && j < len(subString) {
//				if mainString[i] == subString[j] {
//					i, j = i+1, j+1
//				} else if j > 0 {
//					j = next[j]
//				} else {
//					i++
//				}
//			}
//			if len(subString) == j {
//				return i - j
//			} else {
//				return -1
//			}
//		}
//	}
func repeatedSubstringPattern(s string) bool {
	if len(s) == 1 {
		return false
	} else {
		for length := 1; length <= len(s)/2; length++ {
			flag := true
			i := length
			for ; length+i <= len(s); i += length {
				if s[i:length+i] != s[:length] {
					flag = false
					break
				}
			}
			if flag && i == len(s) {
				return flag
			}
		}
		return false
	}
}

//	func removeDuplicates(s string) string {
//		stack := make([]byte, 0, 20000)
//		for i := 0; i < len(s); i++ {
//			if len(stack) == 0 || stack[len(stack)-1] != s[i] {
//				stack = append(stack, s[i])
//			} else if stack[len(stack)-1] == s[i] {
//				stack = stack[:len(stack)-1]
//			}
//		}
//		return string(stack)
//	}
func evalRPN(tokens []string) int {
	stack := make([]string, 0, 100)
	for _, token := range tokens {
		if strings.Contains("+-*/", token) {
			numbers := stack[len(stack)-2:]
			stack = stack[:len(stack)-2]
			num1, _ := strconv.Atoi(numbers[0])
			num2, _ := strconv.Atoi(numbers[1])
			if token == "+" {
				stack = append(stack, strconv.Itoa(num1+num2))
			} else if token == "-" {
				stack = append(stack, strconv.Itoa(num1-num2))
			} else if token == "*" {
				stack = append(stack, strconv.Itoa(num1*num2))
			} else {
				stack = append(stack, strconv.Itoa(num1/num2))
			}
		} else {
			stack = append(stack, token)
		}
	}
	ans, _ := strconv.Atoi(stack[0])
	return ans
}

//	func repeatedStringMatch(a string, b string) int {
//		ab := []byte(a)
//		count := 1
//		for {
//			if indexOf(string(ab), b) != -1 {
//				return count
//			} else {
//				if len(ab) > len(b) {
//					ab = append(ab, []byte(a)...)
//					count++
//					if indexOf(string(ab), b) == -1 {
//						return -1
//					}
//				} else {
//					ab = append(ab, []byte(a)...)
//					count++
//					continue
//				}
//			}
//		}
//	}
func repeatedStringMatch(a string, b string) int {
	k := len(b)/len(a) + 2
	ab, t := []byte(a), []byte(a)
	for i := 1; i <= k; i++ {
		if strings.Index(string(ab), b) != -1 {
			return i
		} else {
			ab = append(ab, t...)
			continue
		}
	}
	return -1
}

//	func longestPalindrome(s string) int {
//		charCount := make(map[string]int)
//		for i := 0; i < len(s); i++ {
//			if _, ok := charCount[s[i:i+1]]; ok {
//				charCount[s[i:i+1]]++
//			} else {
//				charCount[s[i:i+1]] = 1
//			}
//		}
//		length := 0
//		flag := false
//		for _, count := range charCount {
//			if count%2 == 0 {
//				length += count
//			} else {
//				flag = true
//				length += count - 1
//			}
//		}
//		if flag {
//			length++
//		}
//		return length
//	}
func maxSlidingWindow(nums []int, k int) []int {
	ans := make([]int, 0, len(nums)-k+1)
	queue := make([]int, 0)
	for i, num := range nums {
		if len(queue) > 0 && i-queue[0] >= k {
			queue = queue[1:]
		}
		for len(queue) > 0 && nums[queue[len(queue)-1]] < num {
			queue = queue[:len(queue)-1]
		}
		queue = append(queue, i)
		if i >= k-1 {
			ans = append(ans, nums[queue[0]])
		}
	}
	return ans
}

//	func finalPrices(prices []int) (ans []int) {
//		for i := 0; i < len(prices); i++ {
//			j := i + 1
//			for ; j < len(prices); j++ {
//				if prices[j] <= prices[i] {
//					ans = append(ans, prices[i]-prices[j])
//					break
//				}
//			}
//			if j >= len(prices) {
//				ans = append(ans, prices[i])
//			}
//		}
//		return
//	}
func maxAscendingSum(nums []int) int {
	maxSum := nums[0]
	sum := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			sum += nums[i]
			maxSum = max(maxSum, sum)
		} else {
			sum = nums[i]
		}
	}
	return maxSum
}
func interpret(command string) string {
	return strings.ReplaceAll(strings.ReplaceAll(command, "()", "o"), "(al)", "al")
}
func maximizeSum(nums []int, k int) int {
	return (k * (slices.Max(nums)<<1 + k - 1)) >> 1
}
func CheckPermutation(s1 string, s2 string) bool {
	if len(s1) != len(s2) {
		return false
	} else {
		set1 := make(map[string]int)
		set2 := make(map[string]int)
		for i := 0; i < len(s1); i++ {
			if _, ok := set1[s1[i:i+1]]; ok {
				set1[s1[i:i+1]]++
			} else {
				set1[s1[i:i+1]] = 1
			}
			if _, ok := set2[s2[i:i+1]]; ok {
				set2[s2[i:i+1]]++
			} else {
				set2[s2[i:i+1]] = 1
			}
		}
		for k, v1 := range set1 {
			if v2, ok := set2[k]; ok {
				if v1 != v2 {
					return false
				}
			} else {
				return false
			}
		}
		return true
	}
}
func removeDuplicates(nums []int) int {
	i := 0
	for ; i < len(nums); i++ {
		next := i + 1
		for next < len(nums) && nums[next] == nums[i] {
			next++
		}
		if next-i > 2 {
			nums = append(nums[:i+2], nums[next:]...)
		}
	}
	return i
}

//	func longestAlternatingSubarray(nums []int, threshold int) int {
//		maxLen := 0
//		for left := 0; left < len(nums)-maxLen; left++ {
//			for left < len(nums)-maxLen && (nums[left] > threshold || nums[left]%2 != 0) {
//				left++
//			}
//			if left == len(nums)-maxLen {
//				break
//			}
//			right := left
//			for right+1 < len(nums) && nums[right+1] <= threshold && (nums[right+1]+right-left+1)%2 == 0 {
//				right++
//			}
//			maxLen = max(maxLen, right-left+1)
//		}
//		return maxLen
//	}
func longestAlternatingSubarray(nums []int, threshold int) int {
	maxLen := 0
	length := len(nums)
	for left := 0; left < length-maxLen; left++ {
		for left < len(nums)-maxLen && (nums[left] > threshold || nums[left]%2 != 0) {
			left++
		}
		if left == len(nums)-maxLen {
			break
		}
		right := left
		for right+1 < len(nums) && nums[right+1] <= threshold && (nums[right+1]+right-left+1)%2 == 0 {
			right++
		}
		maxLen = max(maxLen, right-left+1)
	}
	return maxLen
}
func distanceTraveled(mainTank int, additionalTank int) int {
	addTimes := min(mainTank/5, additionalTank)
	totalDist := (mainTank + addTimes) * 10
	if (mainTank+addTimes)/5 >= addTimes && additionalTank-addTimes > 0 {
		totalDist += 10
	}
	return totalDist
}

type Man struct {
	Name string
	Sex  string
}
type Superman struct {
	Man
	Level int
	Skill string
}

//	func maximumSum(nums []int) int {
//		set := make(map[int][]int, 100)
//		for _, num := range nums {
//			digitSum := getDigitsSum(num)
//			set[digitSum] = append(set[digitSum], num)
//		}
//		maxDigitSum := -1
//		for _, v := range set {
//			if len(v) >= 2 {
//				maximum, secondMaximum := 0, 0
//				for _, num := range v {
//					if num > maximum {
//						secondMaximum = maximum
//						maximum = num
//					} else if num > secondMaximum {
//						secondMaximum = num
//					}
//				}
//				maxDigitSum = max(maxDigitSum, maximum+secondMaximum)
//			}
//		}
//		return maxDigitSum
//	}
func getMaxArray(prices []int) []int {
	maxPrice, maxArray := 0, make([]int, len(prices))
	for i := len(prices) - 1; i >= 0; i-- {
		maxPrice = max(maxPrice, prices[i])
		maxArray[i] = maxPrice
	}
	return maxArray
}

//	func maxProfit(prices []int) int {
//		maxArray := getMaxArray(prices)
//		for i := 0; i < len(prices); i++ {
//			maxArray[i] -= prices[i]
//		}
//		return slices.Max(maxArray)
//	}
func canConstruct(ransomNote string, magazine string) bool {
	mapRansomNote := make(map[int32]int, 26)
	mapMagazine := make(map[int32]int, 26)
	for _, char := range ransomNote {
		mapRansomNote[char]++
	}
	for _, char := range magazine {
		mapMagazine[char]++
	}
	for k, v := range mapRansomNote {
		if v > mapMagazine[k] {
			return false
		}
	}
	return true
}
func wordPattern(pattern string, s string) bool {
	words := strings.Split(s, " ")
	if len(pattern) != len(words) {
		return false
	} else {
		pattern2WordMap := make(map[int32]string, 26)
		word2PatternMap := make(map[string]int32, 26)
		for i, char := range pattern {
			if _, ok := pattern2WordMap[char]; ok {
				if pattern2WordMap[char] != words[i] {
					return false
				}
			} else {
				if _, ok = word2PatternMap[words[i]]; ok {
					return false
				} else {
					pattern2WordMap[char] = words[i]
					word2PatternMap[words[i]] = char
				}
			}
		}
		return true
	}
}

func findClosestNumber(nums []int) int {
	minDist := Abs(nums[0])
	maxValue := nums[0]
	for i := 1; i < len(nums); i++ {
		if Abs(nums[i]) < minDist {
			minDist = Abs(nums[i])
			maxValue = nums[i]
		} else if Abs(nums[i]) == minDist {
			maxValue = max(maxValue, nums[i])
		}
	}
	return maxValue
}

//	func getDigitsSum(num int) (digitSum int) {
//		for ; num > 0; num /= 10 {
//			digitSum += num % 10
//		}
//		return
//	}
func getDigitsSum(numStr string) string {
	sum := 0
	for i := 0; i < len(numStr); i++ {
		digit, _ := strconv.Atoi(numStr[i : i+1])
		sum += digit
	}
	return strconv.Itoa(sum)
}

//	func digitSum(s string, k int) string {
//		builder := strings.Builder{}
//		for len(s) > k {
//			builder.Reset()
//			for i := 0; i < len(s); i += k {
//				if i+k >= len(s) {
//					builder.Write([]byte(getDigitsSum(s[i:])))
//				} else {
//					builder.Write([]byte(getDigitsSum(s[i : i+k])))
//				}
//			}
//			s = builder.String()
//		}
//		return s
//	}
func inter(nums1 []int, nums2 []int) (inter []int) {
	for _, num1 := range nums1 {
		if slices.Contains(nums2, num1) {
			inter = append(inter, num1)
		}
	}
	return
}
func intersection(nums [][]int) []int {
	ans := nums[0]
	for i := 1; i < len(nums); i++ {
		ans = inter(ans, nums[i])
	}
	slices.Sort(ans)
	return ans
}
func countPrefixes(words []string, s string) (count int) {
	for _, word := range words {
		if strings.HasPrefix(s, word) {
			count++
		}
	}
	return count
}

//func maxSubArray(nums []int) int {
//	length := len(nums)
//	if length == 1 {
//		return nums[0]
//	} else {
//		maxArray := make([]int, len(nums))
//		currentMax := nums[length-1]
//		for i := length - 1; i >= 0; i-- {
//			if i == length-1 || maxArray[i+1] <= 0 {
//				maxArray[i] = nums[i]
//			} else if maxArray[i+1] > 0 {
//				maxArray[i] = nums[i] + maxArray[i+1]
//			}
//			currentMax = max(currentMax, maxArray[i])
//		}
//		return currentMax
//	}
//}

func getSquareDigitSum(n int) int {
	sum := 0
	for n > 0 {
		sum += (n % 10) * (n % 10)
		n /= 10
	}
	return sum
}
func isHappy(n int) bool {
	cache := make([]int, 0, 100)
	for n > 1 {
		n = getSquareDigitSum(n)
		if slices.Contains(cache, n) {
			return false
		} else {
			cache = append(cache, n)
		}
	}
	return true
}
func summaryRanges(nums []int) (ans []string) {
	if len(nums) == 0 {
		return
	} else if len(nums) == 1 {
		ans = append(ans, strconv.Itoa(nums[0]))
		return
	} else {
		left := 0
		for i := 1; i < len(nums); i++ {
			if nums[i]-nums[i-1] != 1 {
				if i-1 == left {
					ans = append(ans, strconv.Itoa(nums[left]))
				} else {
					ans = append(ans, fmt.Sprintf("%d->%d", nums[left], nums[i-1]))
				}
				left = i
				if i == len(nums)-1 {
					ans = append(ans, strconv.Itoa(nums[i]))
				}
			} else {
				if i == len(nums)-1 {
					ans = append(ans, fmt.Sprintf("%d->%d", nums[left], nums[i]))
				}
			}
		}
		return
	}
}

func isPowerOfTwo(n int) bool {
	if n == 0 {
		return false
	}
	return n == (n & (^n + 1))
}

func minDeletion(nums []int) (count int) {
	if len(nums) == 1 {
		return 1
	} else {
		for i := 0; i < len(nums); {
			if (i-count)%2 != 0 {
				i++
			} else {
				k := i
				for k+1 < len(nums) && nums[k] == nums[k+1] {
					k++
				}
				count += k - i
				i = k + 1
			}
		}
		if (len(nums)-count)%2 != 0 {
			count++
		}
		return count
	}
}
func kItemsWithMaximumSum(numOnes int, numZeros int, numNegOnes int, k int) int {
	if numOnes >= k {
		return k
	} else if numOnes+numZeros >= k {
		return numOnes
	} else {
		return 2*numOnes + numZeros - k
	}
}
func pickGifts(gifts []int, k int) (ans int64) {
	length := len(gifts)
	slices.Sort(gifts)
	for i := 0; i < k; i++ {
		sqrt := int(math.Sqrt(float64(gifts[length-1])))
		index := sort.Search(length-1, func(i int) bool {
			return gifts[i] >= sqrt
		})
		gifts = append(gifts[:index], append([]int{sqrt}, gifts[index:length-1]...)...)
	}
	for _, gift := range gifts {
		ans += int64(gift)
	}
	return
}
func addDigits(num int) int {
	for num >= 10 {
		sum := 0
		for num > 0 {
			sum += num % 10
			num /= 10
		}
		num = sum
	}
	return num
}
func checkRecord(s string) bool {
	return strings.Count(s, "A") < 2 && !strings.Contains(s, "LLL")
}
func findErrorNums(nums []int) (ans []int) {
	set := make(map[int]struct{})
	sum := (len(nums) * (len(nums) + 1)) / 2
	actualSum := 0
	for _, num := range nums {
		actualSum += num
		if _, ok := set[num]; ok {
			ans = append(ans, num)
		} else {
			set[num] = struct{}{}
		}
	}
	ans = append(ans, sum-actualSum+ans[0])
	return
}
func entityParser(text string) string {
	text = strings.ReplaceAll(text, "&quot;", "\"")
	text = strings.ReplaceAll(text, "&apos;", "'")
	text = strings.ReplaceAll(text, "&gt;", ">")
	text = strings.ReplaceAll(text, "&lt;", "<")
	text = strings.ReplaceAll(text, "&frasl;", "/")
	text = strings.ReplaceAll(text, "&amp;", "&")
	strings.NewReplacer()
	return text
}
func maximumStrongPairXor(nums []int) int {
	if len(nums) == 1 {
		return nums[0] ^ nums[0]
	} else {
		maxXOR := 0
		for i := 0; i < len(nums); i++ {
			for j := i; j < len(nums); j++ {
				if Abs(nums[i]-nums[j]) <= min(nums[i], nums[j]) {
					maxXOR = max(maxXOR, nums[i]^nums[j])
				}
			}
		}
		return maxXOR
	}
}

type MyLinkedList struct {
	size int
	head *ListNode
}

//func Constructor() MyLinkedList {
//	return MyLinkedList{size: 0, head: &ListNode{Val: -1, Next: nil}}
//}

func (this *MyLinkedList) Get(index int) int {
	if index >= this.size {
		return -1
	} else {
		p := this.head
		for i := 0; i <= index; i++ {
			p = p.Next
		}
		return p.Val
	}
}

func (this *MyLinkedList) AddAtHead(val int) {
	this.head.Next = &ListNode{Val: val, Next: this.head.Next}
	this.size++
}

func (this *MyLinkedList) AddAtTail(val int) {
	p := this.head
	for p.Next != nil {
		p = p.Next
	}
	p.Next = &ListNode{Val: val, Next: nil}
	this.size++
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {
	if index == this.size {
		this.AddAtTail(val)
	} else if index < this.size {
		p := this.head
		for i := 0; i < index; i++ {
			p = p.Next
		}
		p.Next = &ListNode{Val: val, Next: p.Next}
		this.size++
	}
}

func (this *MyLinkedList) DeleteAtIndex(index int) {
	if index < this.size {
		p := this.head
		for i := 0; i < index; i++ {
			p = p.Next
		}
		p.Next = p.Next.Next
	}
	this.size--
}
func maximumTime(time string) string {
	builder := strings.Builder{}
	if time[0] == '?' {
		if time[1] != '?' && time[1] >= '4' {
			builder.WriteByte('1')
		} else {
			builder.WriteByte('2')
		}
	} else {
		builder.WriteByte(time[0])
	}
	if time[1] == '?' {
		if time[0] == '0' || time[0] == '1' {
			builder.WriteByte('9')
		} else {
			builder.WriteByte('3')
		}
	} else {
		builder.WriteByte(time[1])
	}
	builder.WriteByte(':')
	if time[3] == '?' {
		builder.WriteByte('5')
	} else {
		builder.WriteByte(time[3])
	}
	if time[4] == '?' {
		builder.WriteByte('9')
	} else {
		builder.WriteByte(time[4])
	}
	return builder.String()
}
func diagonalSum(mat [][]int) (sum int) {
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			if i == j || i+j == len(mat)-1 {
				sum += mat[i][j]
			}
		}
	}
	return
}
func isAlienSorted(words []string, order string) bool {
	dictOrder := make(map[byte]int)
	for i, char := range order {
		dictOrder[byte(char)] = i
	}
	newWords := make([]string, len(words))
	copy(newWords, words)
	slices.SortFunc(newWords, func(a, b string) int {
		length := min(len(a), len(b))
		for i := 0; i < length; i++ {
			if a[i] != b[i] {
				return dictOrder[a[i]] - dictOrder[b[i]]
			}
		}
		return len(a) - len(b)
	})
	for i := 0; i < len(words); i++ {
		if newWords[i] != words[i] {
			return false
		}
	}
	return true
}

type myType struct {
	index int
	char  byte
}

func restoreString(s string, indices []int) string {
	myTypeSlice := make([]myType, 0, len(s))
	for i := 0; i < len(s); i++ {
		myTypeSlice = append(myTypeSlice, myType{index: indices[i], char: s[i]})
	}
	slices.SortFunc(myTypeSlice, func(a, b myType) int {
		return a.index - b.index
	})
	builder := strings.Builder{}
	for i := 0; i < len(s); i++ {
		builder.WriteByte(myTypeSlice[i].char)
	}
	return builder.String()
}
func arithmeticTriplets(nums []int, diff int) (count int) {
	numCount := make(map[int]int)
	for _, num := range nums {
		numCount[num] += 1
	}
	for _, num := range nums {
		count += numCount[num] * numCount[num+diff] * numCount[num+2*diff]
	}
	return count
}
func numberOfMatches(n int) (count int) {
	for n > 1 {
		count += n / 2
		n -= n / 2
	}
	return
}

//	func maxScore(cardPoints []int, k int) int {
//		n := len(cardPoints)
//		score := 0
//		for i := 0; i < n-k; i++ {
//			score += cardPoints[i]
//		}
//		sum := score
//		for i := n - k; i < n; i++ {
//			sum += cardPoints[i]
//		}
//		minScore := score
//		for i := 1; i <= k; i++ {
//			score = score - cardPoints[i-1] + cardPoints[n-k+i-1]
//			minScore = min(minScore, score)
//		}
//		return sum - minScore
//	}
func stringMatching(words []string) (ans []string) {
	slices.SortFunc(words, func(a, b string) int {
		return len(a) - len(b)
	})
	set := make(map[string]struct{})
	for i := 0; i < len(words)-1; i++ {
		for j := i + 1; j < len(words); j++ {
			if strings.Contains(words[j], words[i]) {
				if _, ok := set[words[i]]; !ok {
					set[words[i]] = struct{}{}
				}
			}
		}
	}
	for k, _ := range set {
		ans = append(ans, k)
	}
	return
}
func busyStudent(startTime []int, endTime []int, queryTime int) (count int) {
	length := len(startTime)
	for i := 0; i < length; i++ {
		if startTime[i] <= queryTime && endTime[i] >= queryTime {
			count += 1
		}
	}
	return
}
func checkXMatrix(grid [][]int) bool {
	n := len(grid)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j || i+j == n-1 {
				if grid[i][j] == 0 {
					return false
				}
			} else {
				if grid[i][j] != 0 {
					return false
				}
			}
		}
	}
	return true
}
func average(salary []int) float64 {
	sum := 0
	maxSalary := 0
	minSalary := salary[0]
	for _, s := range salary {
		sum += s
		if s > maxSalary {
			maxSalary = s
		} else if s < minSalary {
			minSalary = s
		}
	}
	return float64(sum-maxSalary-minSalary) / float64(len(salary)-2)
}
func carPooling(trips [][]int, capacity int) bool {
	upDown := make([]int, 1001)
	for _, trip := range trips {
		upDown[trip[1]] += trip[0]
		upDown[trip[2]] -= trip[0]
	}
	for _, num := range upDown {
		capacity -= num
		if capacity < 0 {
			return false
		}
	}
	return true
}
func findJudge(n int, trust [][]int) int {
	//信任人的数量 被信任的数量 要求仅有一个
	trustMap := make([][2]int, n)
	for _, t := range trust {
		trustMap[t[0]-1][0] += 1
		trustMap[t[1]-1][1] += 1
	}
	judge := -1
	for i, t := range trustMap {
		if t[0] == 0 && t[1] == n-1 {
			if judge != -1 {
				return -1
			} else {
				judge = i
			}
		}
	}
	if judge != -1 {
		return judge + 1
	} else {
		return -1
	}
}
