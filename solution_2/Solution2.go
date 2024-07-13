package main

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

//	func distributeCandies(n int, limit int) (count int) {
//		for i := 0; i <= limit; i++ {
//			for j := 0; j <= limit; j++ {
//				for k := 0; k <= limit; k++ {
//					if i+j+k == n {
//						count++
//					}
//				}
//			}
//		}
//		return
//	}
func countPairs(nums []int, target int) (count int) {
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[i]+nums[j] < target {
				count++
			}
		}
	}
	return
}

//	type ListNode struct {
//		Val  int
//		Next *ListNode
//	}
//
//	type DualListNode struct {
//		Val  int
//		Next *DualListNode
//		Pre  *DualListNode
//	}
//
//	type FrontMiddleBackQueue struct {
//		head   *DualListNode
//		tail   *DualListNode
//		middle *DualListNode
//	}
//
//	func Constructor() FrontMiddleBackQueue {
//		head := &DualListNode{Val: 0, Next: nil, Pre: nil}
//		return FrontMiddleBackQueue{head: head, tail: nil, middle: nil}
//	}
//
//	func (this *FrontMiddleBackQueue) PushFront(val int) {
//		newNode := &DualListNode{Val: val, Next: this.head.Next, Pre: this.head}
//		if this.head.Next != nil {
//			this.head.Next.Pre = newNode
//		}
//		this.head.Next = newNode
//		if this.tail == nil {
//			this.tail = newNode
//		}
//		if this.middle == nil {
//			this.middle = newNode
//		}
//
//		this.head.Val += 1
//
//		if this.head.Val%2 == 0 {
//			this.middle = this.middle.Pre
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PushMiddle(val int) {
//		if this.middle == nil {
//			this.PushFront(val)
//		} else {
//			if this.head.Val%2 == 0 {
//				newNode := &DualListNode{Val: val, Pre: this.middle, Next: this.middle.Next}
//				if this.middle.Next != nil {
//					this.middle.Next.Pre = newNode
//				}
//				this.middle.Next = newNode
//				this.head.Val += 1
//				if this.head.Val%2 != 0 {
//					this.middle = this.middle.Next
//				}
//			} else {
//				p := this.middle.Pre
//				newNode := &DualListNode{Val: val, Pre: p, Next: this.middle}
//				this.middle.Pre = newNode
//				p.Next = newNode
//				this.head.Val += 1
//				if this.head.Val%2 == 0 {
//					this.middle = this.middle.Pre
//				}
//			}
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PushBack(val int) {
//		if this.tail == nil {
//			this.PushFront(val)
//		} else {
//			newNode := &DualListNode{Val: val, Pre: this.tail, Next: nil}
//			this.tail.Next = newNode
//			this.tail = newNode
//			this.head.Val += 1
//			if this.head.Val%2 != 0 {
//				this.middle = this.middle.Next
//			}
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PopFront() int {
//		if this.head.Val == 0 {
//			return -1
//		} else {
//			val := this.head.Next.Val
//			this.head.Next = this.head.Next.Next
//			if this.head.Val%2 == 0 {
//				this.middle = this.middle.Next
//			} else if this.head.Val == 1 {
//				this.middle = nil
//				this.tail = nil
//			}
//			if this.head.Next != nil {
//				this.head.Next.Pre = this.head
//			}
//			this.head.Val -= 1
//			return val
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PopMiddle() int {
//		if this.head.Val == 0 {
//			return -1
//		} else {
//			val := this.middle.Val
//			p := this.middle.Pre
//			p.Next = this.middle.Next
//			if p.Next != nil {
//				p.Next.Pre = p
//			}
//			if this.head.Val%2 == 0 {
//				this.middle = this.middle.Next
//			} else {
//				this.middle = this.middle.Pre
//			}
//			if this.head.Val == 1 {
//				this.middle = nil
//				this.tail = nil
//			}
//			this.head.Val -= 1
//			return val
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PopBack() int {
//		if this.head.Val == 0 {
//			return -1
//		} else {
//			val := this.tail.Val
//			this.tail = this.tail.Pre
//			this.tail.Next = nil
//			if this.head.Val%2 != 0 {
//				this.middle = this.middle.Pre
//			}
//			if this.head.Val == 1 {
//				this.middle = nil
//				this.tail = nil
//			}
//			this.head.Val -= 1
//			return val
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PrintQueue() {
//		for p := this.head.Next; p != nil; p = p.Next {
//			fmt.Print(p.Val, " ")
//		}
//	}

type FrontMiddleBackQueue struct {
	queue []int
}

//func Constructor() FrontMiddleBackQueue {
//	return FrontMiddleBackQueue{queue: make([]int, 0, 1000)}
//}

func (this *FrontMiddleBackQueue) PushFront(val int) {
	this.queue = append([]int{val}, this.queue...)
}

func (this *FrontMiddleBackQueue) PushMiddle(val int) {
	length := len(this.queue)
	this.queue = append(this.queue[:length/2], append([]int{val}, this.queue[length/2:]...)...)
}

func (this *FrontMiddleBackQueue) PushBack(val int) {
	this.queue = append(this.queue, val)
}

//	func (this *FrontMiddleBackQueue) PopFront() int {
//		if this.IsEmpty() {
//			return -1
//		} else {
//			val := this.queue[0]
//			this.queue = this.queue[1:]
//			return val
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PopMiddle() int {
//		if this.IsEmpty() {
//			return -1
//		} else {
//			length := len(this.queue)
//			if length%2 == 0 {
//				val := this.queue[length/2-1]
//				this.queue = append(this.queue[:length/2-1], this.queue[length/2:]...)
//				return val
//			} else {
//				val := this.queue[length/2]
//				this.queue = append(this.queue[:length/2], this.queue[length/2+1:]...)
//				return val
//			}
//		}
//	}
//
//	func (this *FrontMiddleBackQueue) PopBack() int {
//		if this.IsEmpty() {
//			return -1
//		} else {
//			val := this.queue[len(this.queue)-1]
//			this.queue = this.queue[:len(this.queue)-1]
//			return val
//		}
//	}
func (this *FrontMiddleBackQueue) IsEmpty() bool {
	return len(this.queue) == 0
}

type SmallestInfiniteSet struct {
	min      int
	notInMap map[int]bool
}

//
//func Constructor() SmallestInfiniteSet {
//	return SmallestInfiniteSet{min: 1, notInMap: make(map[int]bool)}
//}

func (this *SmallestInfiniteSet) PopSmallest() int {
	val := this.min
	this.notInMap[val] = true
	for i := val; ; i++ {
		if !this.notInMap[i] {
			this.min = i
			break
		}
	}
	return val
}

func (this *SmallestInfiniteSet) AddBack(num int) {
	if this.notInMap[num] {
		this.notInMap[num] = false
		this.min = min(this.min, num)
	}
}

func closeStrings(word1 string, word2 string) bool {
	if len(word1) == len(word2) {
		countMap1 := make(map[uint8]int)
		countMap2 := make(map[uint8]int)
		for i := 0; i < len(word1); i++ {
			countMap1[word1[i]] += 1
			countMap2[word2[i]] += 1
		}
		arr1 := make([]int, 0, 26)
		for k, v := range countMap1 {
			if countMap2[k] == 0 {
				return false
			} else {
				arr1 = append(arr1, v)
			}
		}
		arr2 := make([]int, 0, 26)
		for _, v := range countMap2 {
			arr2 = append(arr2, v)
		}
		slices.Sort(arr1)
		slices.Sort(arr2)
		return slices.Equal(arr1, arr2)
	}
	return false
}
func countWords(words1 []string, words2 []string) (count int) {
	countMap1 := make(map[string]int, 1000)
	countMap2 := make(map[string]int, 1000)
	for _, word := range words1 {
		countMap1[word] += 1
	}
	for _, word := range words2 {
		countMap2[word] += 1
	}
	for k, v := range countMap1 {
		if v == 1 && countMap2[k] == 1 {
			count++
		}
	}
	return
}
func areSimilar(mat [][]int, k int) bool {
	for i, m := range mat {
		width := len(m)
		if i%2 == 0 {
			if !slices.Equal(m, append(m[k%width:], m[:k%width]...)) {
				return false
			}
		} else {
			if !slices.Equal(m, append(m[width-k%width:], m[:width-k%width]...)) {
				return false
			}
		}
	}
	return true
}
func largeGroupPositions(s string) (ans [][]int) {
	for i := 0; i <= len(s)-2; {
		j := i + 1
		for ; j < len(s); j++ {
			if s[j] != s[i] {
				if j-i >= 3 {
					ans = append(ans, []int{i, j - 1})
				}
				break
			}
		}
		if j == len(s) && j-i >= 3 {
			ans = append(ans, []int{i, j - 1})
			break
		}
		i = j
	}
	return
}

func maskPII(s string) string {
	if strings.Contains(s, "@") {
		name := strings.Split(s, "@")[0]
		name = strings.ToLower(name[:1] + "*****" + name[len(name)-1:])
		domain := strings.ToLower(strings.Split(s, "@")[1])
		return name + "@" + domain
	} else {
		phone := make([]byte, 0)
		for i := 0; i < len(s); i++ {
			if !strings.Contains("+-() ", s[i:i+1]) {
				phone = append(phone, s[i])
			}
		}
		fmt.Println(string(phone))
		phoneArr := make([]byte, 0)
		for i := len(phone) - 1; i >= 0; i-- {
			if len(phoneArr) < 4 {
				phoneArr = append([]byte{phone[i]}, phoneArr...)
			} else {
				if len(phoneArr) == 4 || len(phoneArr) == 8 || (len(phoneArr) == 12) {
					phoneArr = append([]byte{'-'}, phoneArr...)
				}
				phoneArr = append([]byte{'*'}, phoneArr...)
			}
		}
		if len(phoneArr) > 12 {
			phoneArr = append([]byte{'+'}, phoneArr...)
		}
		return string(phoneArr)
	}
}

//	func checkRow(mat [][]int, i int) bool {
//		for j := 0; j < len(mat[i]); j++ {
//			if mat[i][j] != -1 {
//				return false
//			}
//		}
//		return true
//	}
//
//	func checkColumn(mat [][]int, j int) bool {
//		for i := 0; i < len(mat); i++ {
//			if mat[i][j] != -1 {
//				return false
//			}
//		}
//		return true
//	}
func firstCompleteIndex(arr []int, mat [][]int) int {
	m, n := len(mat), len(mat[0])
	locationMap := make(map[int][2]int, m)
	for i, row := range mat {
		for j, num := range row {
			locationMap[num] = [2]int{i, j}
		}
	}
	rowColSum := make([]int, m+n)
	for index, num := range arr {
		i := locationMap[num][0]
		j := locationMap[num][1]
		rowColSum[i] += 1
		rowColSum[j+m] += 1
		if rowColSum[i] == n || rowColSum[j+m] == m {
			return index
		}
	}
	return -1
}
func defangIPaddr(address string) string {
	return strings.ReplaceAll(address, ".", "[.]")
}
func runningSum(nums []int) (ans []int) {
	sum := 0
	for _, num := range nums {
		sum += num
		ans = append(ans, sum)
	}
	return
}
func findLucky(arr []int) int {
	countMap := make(map[int]int)
	for _, num := range arr {
		countMap[num] += 1
	}
	maxLuckyNumber := -1
	for k, v := range countMap {
		if k == v {
			maxLuckyNumber = max(maxLuckyNumber, k)
		}
	}
	return maxLuckyNumber
}
func shuffle(nums []int, n int) []int {
	newNums := make([]int, 0, 2*n)
	for i := 0; i < n; i++ {
		newNums = append(newNums, nums[i], nums[i+n])
	}
	return newNums
}

//	func maxProduct(nums []int) int {
//		slices.Sort(nums)
//		length := len(nums)
//		return (nums[length-1] - 1) * (nums[length-2] - 1)
//	}
func isPrefixOfWord(sentence string, searchWord string) int {
	words := strings.Split(strings.TrimSpace(sentence), " ")
	for i, word := range words {
		if strings.HasPrefix(word, searchWord) {
			return i + 1
		}
	}
	return -1
}
func countElements(nums []int) (count int) {
	maxNum, minNum := nums[0], nums[0]
	for _, num := range nums {
		maxNum = max(maxNum, num)
		minNum = min(minNum, num)
	}
	for _, num := range nums {
		if num < maxNum && num > minNum {
			count++
		}
	}
	return
}

type MyCircularQueue struct {
	limit int
	size  int
	head  int
	rear  int
	queue []int
}

//func Constructor(k int) MyCircularQueue {
//	return MyCircularQueue{
//		limit: k,
//		queue: make([]int, k),
//	}
//}

func (this *MyCircularQueue) EnQueue(value int) bool {
	if this.size < this.limit {
		this.queue[this.rear] = value
		this.size++
		this.rear = (this.rear + 1) % this.limit
		return true
	}
	return false
}
func (this *MyCircularQueue) DeQueue() bool {
	if this.size > 0 {
		this.size--
		this.head = (this.head + 1) % this.limit
		return true
	}
	return false
}

func (this *MyCircularQueue) Front() int {
	if this.IsEmpty() {
		return -1
	} else {
		return this.queue[this.head]
	}
}

func (this *MyCircularQueue) Rear() int {
	if this.IsEmpty() {
		return -1
	} else {
		rear := (this.rear - 1 + this.limit) % this.limit
		return this.queue[rear]
	}
}

func (this *MyCircularQueue) IsEmpty() bool {
	return this.size == 0
}

func (this *MyCircularQueue) IsFull() bool {
	return this.size == this.limit
}

type DataStream struct {
	value      int
	k          int
	data       []int
	lastStatus bool
}

//func Constructor(value int, k int) DataStream {
//	return DataStream{
//		value:      value,
//		k:          k,
//		data:       make([]int, 0),
//		lastStatus: false,
//	}
//}

func (this *DataStream) Consec(num int) bool {
	this.data = append(this.data, num)
	length := len(this.data)
	if length < this.k {
		return false
	} else {
		if this.lastStatus == true {
			return num == this.value
		} else {
			return check(this.data[length-this.k:], this.value)
		}
	}
}
func check(data []int, value int) bool {
	for _, num := range data {
		if num != value {
			return false
		}
	}
	return true
}
func countTriples(n int) int {
	count := 0
	for i := 1; i < n; i++ {
		for j := 1; j < n; j++ {
			squareSum := i*i + j*j
			sqrt := int(math.Sqrt(float64(squareSum)))
			if (sqrt*sqrt == squareSum && sqrt <= n) || ((sqrt+1)*(sqrt+1) == squareSum && sqrt+1 <= n) {
				count++
			}
		}
	}
	return count
}
func generateMatrix(n int) [][]int {
	matrix := make([][]int, n)
	for i := range matrix {
		matrix[i] = make([]int, n)
	}
	i := 0
	j := 0
	number := 1
	k := n
	for ; k > n/2; k -= 1 {
		matrix[i][j] = number
		for ; i < k-1; i++ {
			for ; j < k-1; j++ {
				matrix[i][j] = number
				number++
			}
			matrix[i][j] = number
			number++
		}
		for ; i > n-k; i-- {
			for ; j > n-k; j-- {
				matrix[i][j] = number
				number++
			}
			matrix[i][j] = number
			number++
		}
		i++
		j++
	}
	return matrix
}

//func removeElements(head *ListNode, val int) *ListNode {
//	head = &ListNode{Val: -1, Next: head}
//	p := head
//	for p.Next != nil {
//		if p.Next.Val == val {
//			p.Next = p.Next.Next
//		} else {
//			p = p.Next
//		}
//	}
//	return head.Next
//}

//	func removeElements(head *ListNode, val int) *ListNode {
//		for head != nil && head.Val == val {
//			head = head.Next
//		}
//		if head != nil {
//			for p := head; p.Next != nil; {
//				if p.Next.Val == val {
//					p.Next = p.Next.Next
//				} else {
//					p = p.Next
//				}
//			}
//		}
//		return head
//	}
func maxDivScore(nums []int, divisors []int) int {
	maxScore := 0
	maxLoc := 0
	for i, divisor := range divisors {
		score := 0
		for _, num := range nums {
			if num%divisor == 0 {
				score++
			}
		}
		if score > maxScore {
			maxScore = score
			maxLoc = i
		} else if score == maxScore {
			if divisor < divisors[maxLoc] {
				maxLoc = i
			}
		}
	}
	return divisors[maxLoc]
}

//	func maxScore(nums []int) int {
//		slices.Sort(nums)
//		slices.Reverse(nums)
//		sum := 0
//		score := 0
//		for _, num := range nums {
//			sum += num
//			if sum > 0 {
//				score++
//			}
//		}
//		return score
//	}
func constructString(n int) string {
	str := strings.Builder{}
	for i := 0; i < n; i++ {
		str.WriteByte('0')
	}
	for i := 0; i < n; i++ {
		str.WriteByte('1')
	}
	return str.String()
}

func findTheLongestBalancedSubstring(s string) int {
	i := 0
	for strings.Contains(s, constructString(i)) {
		i++
	}
	return (i - 1) * 2
}

//	func isPalindrome(s string) bool {
//		left := 0
//		right := len(s) - 1
//		for left <= right {
//			if s[left:left+1] != s[right:right+1] {
//				return false
//			}
//			left++
//			right--
//		}
//		return true
//	}
//
//	func validPalindrome(s string) bool {
//		left := 0
//		right := len(s) - 1
//		for left <= right {
//			if s[left:left+1] != s[right:right+1] {
//				return isPalindrome(s[left+1:right+1]) || isPalindrome(s[left:right])
//			}
//			left++
//			right--
//		}
//		return true
//	}
func search(nums []int, target int) int {
	l := 0
	r := len(nums) - 1
	for l <= r {
		m := (l + r) / 2
		if target == nums[m] {
			return m
		} else if target > nums[m] {
			l = m + 1
		} else {
			r = m - 1
		}
	}
	return -1
}

//func twoSum(numbers []int, target int) []int {
//	for index1, num := range numbers {
//		index2 := search(numbers[index1+1:], target-num)
//		if index2 != -1 {
//			return []int{index1 + 1, index1 + index2 + 2}
//		}
//	}
//	return []int{-1}
//}

func findRepeatedDnaSequences(s string) []string {
	set := make(map[string]int)
	left := 0
	for ; left+9 < len(s); left = left + 1 {
		if count, ok := set[s[left:left+10]]; ok {
			if count == 1 {
				set[s[left:left+10]] = count + 1
			}
		} else {
			set[s[left:left+10]] = 1
		}
	}
	arr := make([]string, 0, len(set))
	for k, v := range set {
		if v > 1 {
			arr = append(arr, k)
		}
	}
	return arr
}
func arrSum(arr []int) (result int) {
	for _, num := range arr {
		result += num
	}
	return
}
func findChampion(grid [][]int) int {
	for i := 0; i < len(grid)-1; i++ {
		champion := true
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] != 1 && i != j {
				champion = false
			}
		}
		if champion {
			return i
		}
	}
	return -1
}
func minusAbs(num1 int, num2 int) int {
	minus := num1 - num2
	if minus < 0 {
		return -minus
	} else {
		return minus
	}
}
func findTheDistanceValue(arr1 []int, arr2 []int, d int) int {
	count := 0
	for _, num1 := range arr1 {
		ok := true
		for _, num2 := range arr2 {
			if minusAbs(num1, num2) <= d {
				ok = false
				break
			}
		}
		if ok {
			count++
		}
	}
	return count
}
func kidsWithCandies(candies []int, extraCandies int) []bool {
	maxCandy := 0
	ifMaxCandy := make([]bool, len(candies))
	for _, candyNum := range candies {
		maxCandy = max(maxCandy, candyNum)
	}
	for i, candyNum := range candies {
		ifMaxCandy[i] = candyNum+extraCandies >= maxCandy
	}
	return ifMaxCandy
}

type ParkingSystem struct {
	big    int
	medium int
	small  int
}

//func Constructor(big int, medium int, small int) ParkingSystem {
//	return ParkingSystem{big: big, medium: medium, small: small}
//}

func (this *ParkingSystem) AddCar(carType int) bool {
	if carType == 1 {
		if this.big >= 1 {
			this.big--
			return true
		} else {
			return false
		}
	} else if carType == 2 {
		if this.medium >= 1 {
			this.medium--
			return true
		} else {
			return false
		}
	} else {
		if this.small >= 1 {
			this.small--
			return true
		} else {
			return false
		}
	}
}

func circularGameLosers(n int, k int) []int {
	set := map[int]struct{}{0: {}}
	loseArr := make([]int, n-1)
	for i, _ := range loseArr {
		loseArr[i] = i + 1
	}
	cur := 0
	for i := 1; ; i++ {
		cur = (cur + i*k) % n
		if _, ok := set[cur]; ok {
			break
		} else {
			set[cur] = struct{}{}
			loseArr = slices.DeleteFunc(loseArr, func(n int) bool {
				return n == cur
			})
		}
	}
	for i, _ := range loseArr {
		loseArr[i]++
	}
	return loseArr
}

//func lengthOfLongestSubstring(s string) int {
//	if len(s) <= 1 {
//		return len(s)
//	}
//	maxLen := 0
//	left := 0
//	set := make(map[string]struct{})
//	for right := 0; right < len(s); right++ {
//		_, ok := set[s[right:right+1]]
//		for ; ok; _, ok = set[s[right:right+1]] {
//			delete(set, s[left:left+1])
//			left++
//		}
//		set[s[right:right+1]] = struct{}{}
//		maxLen = max(maxLen, right-left+1)
//	}
//	return maxLen
//}

//	func minSubArrayLen(target int, nums []int) int {
//		i := 0
//		sum := 0
//		minLen := len(nums)
//		flag := false
//		for j := 0; j < len(nums); j++ {
//			if sum < target {
//				sum += nums[j]
//			}
//			if sum >= target {
//				flag = true
//				for sum >= target {
//					minLen = min(minLen, j-i+1)
//					sum -= nums[i]
//					i++
//				}
//			}
//		}
//		if flag {
//			return minLen
//		} else {
//			return 0
//		}
//	}
func containsNearbyDuplicate(nums []int, k int) bool {
	set := make(map[int]int)
	for i, num := range nums {
		if j, ok := set[num]; ok {
			if i-j <= k {
				return true
			}
		}
		set[num] = i
	}
	return false
}

func vowelStrings(words []string, left int, right int) (count int) {
	for _, word := range words[left : right+1] {
		if strings.Contains("aeiou", word[:1]) && strings.Contains("aeiou", word[len(word)-1:]) {
			count++
		}
	}
	return
}
func checkString(s string) bool {
	return !strings.Contains(s, "ba")
}

//	func isPalindrome(s string) bool {
//		left := 0
//		right := len(s) - 1
//		s = strings.ToLower(s)
//		for left <= right {
//			for !unicode.IsDigit(rune(s[left])) && !unicode.IsLetter(rune(s[left])) {
//				left++
//				if left > right {
//					return true
//				}
//			}
//			for !unicode.IsDigit(rune(s[right])) && !unicode.IsLetter(rune(s[right])) {
//				right--
//				if left > right {
//					return true
//				}
//			}
//			if s[left:left+1] != s[right:right+1] {
//				return false
//			}
//			left++
//			right--
//		}
//		return true
//	}
