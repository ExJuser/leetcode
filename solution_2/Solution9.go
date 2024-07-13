package main

import (
	"container/heap"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

//func numEquivDominoPairs(dominoes [][]int) int {
//	dominoMap := make(map[int]int)
//	for _, domino := range dominoes {
//		key := (domino[0]+domino[1])*10 + Abs(domino[0]-domino[1])
//		dominoMap[key]++
//	}
//	ans := 0
//	for _, v := range dominoMap {
//		ans += v * (v - 1) / 2
//	}
//	return ans
//}
//func mergeNodes(head *ListNode) *ListNode {
//	val := 0
//	h := head
//	for p := head.Next; p != nil; p = p.Next {
//		if p.Val == 0 {
//			h.Val = val
//			if p.Next != nil {
//				val = 0
//				h = h.Next
//			}
//		} else {
//			val += p.Val
//		}
//	}
//	h.Next = nil
//	return head
//}

func maxCount(banned []int, n int, maxSum int) int {
	slices.Sort(banned)
	banIndex := 0
	curSum := 0
	cnt := 0
	for i := 1; i <= n; i++ {
		if banIndex == len(banned) || i != banned[banIndex] {
			if curSum+i <= maxSum {
				cnt++
				curSum += i
			} else {
				return cnt
			}
		} else {
			for banIndex < len(banned) && banned[banIndex] == i {
				banIndex++
			}
		}
	}
	return cnt
}
func numDifferentIntegers(word string) int {
	intSet := make(map[string]struct{})
	for i := 0; i < len(word); {
		if unicode.IsNumber(rune(word[i])) {
			j := i + 1
			for j < len(word) && unicode.IsNumber(rune(word[j])) {
				j++
			}
			intSet[strings.TrimLeft(word[i:j], "0")] = struct{}{}
			i = j
		} else {
			i++
		}
	}
	return len(intSet)
}
func waysToSplitArray(nums []int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	leftSum := 0
	cnt := 0
	for i := 0; i < len(nums)-1; i++ {
		leftSum += nums[i]
		if leftSum >= sum-leftSum {
			cnt++
		}
	}
	return cnt
}
func fairCandySwap(aliceSizes []int, bobSizes []int) []int {
	hashAlice := make(map[int]struct{})
	hashBob := make(map[int]struct{})
	var sumAlice, sumBob int
	for _, alice := range aliceSizes {
		sumAlice += alice
		if _, ok := hashAlice[alice]; !ok {
			hashAlice[alice] = struct{}{}
		}
	}
	for _, bob := range bobSizes {
		sumBob += bob
		if _, ok := hashBob[bob]; !ok {
			hashBob[bob] = struct{}{}
		}
	}
	diff := (sumBob - sumAlice) / 2
	for i := 1; ; i++ {
		if i+diff > 0 {
			_, okAlice := hashAlice[i]
			_, okBob := hashBob[i+diff]
			if okAlice && okBob {
				return []int{i, i + diff}
			}
		}
	}
}
func asteroidsDestroyed(mass int, asteroids []int) bool {
	slices.Sort(asteroids)
	for i := 0; i < len(asteroids); i++ {
		if mass > asteroids[len(asteroids)-1] {
			return true
		} else {
			if mass >= asteroids[i] {
				mass += asteroids[i]
			} else {
				return false
			}
		}
	}
	return true
}

//type FrequencyHeap [][2]int
//
//func (f FrequencyHeap) Len() int {
//	return len(f)
//}
//
//func (f FrequencyHeap) Less(i, j int) bool {
//	return f[i][1] < f[j][1]
//}
//
//func (f FrequencyHeap) Swap(i, j int) {
//	f[i], f[j] = f[j], f[i]
//}
//
//func (f *FrequencyHeap) Push(x any) {
//	*f = append(*f, x.([2]int))
//}
//
//func (f *FrequencyHeap) Pop() any {
//	x := (*f)[f.Len()-1]
//	*f = (*f)[:f.Len()-1]
//	return x
//}
//func topKFrequent(nums []int, k int) []int {
//	hp := &FrequencyHeap{}
//	count := make(map[int]int)
//	for _, num := range nums {
//		count[num]++
//	}
//	for key, value := range count {
//		heap.Push(hp, [2]int{key, value})
//		if hp.Len() > k {
//			heap.Pop(hp)
//		}
//	}
//	ans := make([]int, 0, k)
//	for i := 0; i < k; i++ {
//		ans = append(ans, heap.Pop(hp).([2]int)[0])
//	}
//	return ans
//}

//func findKthLargest(nums []int, k int) int {
//	hp := &IntHeap{}
//	heap.Init(hp)
//	for _, num := range nums {
//		heap.Push(hp, num)
//		if hp.Len() > k {
//			heap.Pop(hp)
//		}
//	}
//	return heap.Pop(hp).(int)
//}

//type Frequency struct {
//	char byte
//	freq int
//}
//type FrequencyHeap []Frequency
//
//func (f FrequencyHeap) Len() int {
//	return len(f)
//}
//
//func (f FrequencyHeap) Less(i, j int) bool {
//	return f[i].freq > f[j].freq
//}
//
//func (f FrequencyHeap) Swap(i, j int) {
//	f[i], f[j] = f[j], f[i]
//}
//
//func (f *FrequencyHeap) Push(x any) {
//	*f = append(*f, x.(Frequency))
//}
//
//func (f *FrequencyHeap) Pop() any {
//	x := (*f)[f.Len()-1]
//	*f = (*f)[:f.Len()-1]
//	return x
//}
//
//func frequencySort(s string) string {
//	hp := &FrequencyHeap{}
//	count := make(map[byte]int)
//	for _, char := range s {
//		count[byte(char)]++
//	}
//	for k, v := range count {
//		heap.Push(hp, Frequency{char: k, freq: v})
//	}
//	builder := strings.Builder{}
//	for i := 0; i < len(count); i++ {
//		x := heap.Pop(hp).(Frequency)
//		for j := 0; j < x.freq; j++ {
//			builder.WriteByte(x.char)
//		}
//	}
//	return builder.String()
//}

type Frequency struct {
	word string
	freq int
}
type FrequencyHeap []Frequency

func (f FrequencyHeap) Len() int {
	return len(f)
}

func (f FrequencyHeap) Less(i, j int) bool {
	if f[i].freq == f[j].freq {
		return f[i].word < f[j].word
	} else {
		return f[i].freq > f[j].freq
	}
}

func (f FrequencyHeap) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

func (f *FrequencyHeap) Push(x any) {
	*f = append(*f, Frequency{word: x.(Frequency).word, freq: x.(Frequency).freq})
}

func (f *FrequencyHeap) Pop() any {
	x := (*f)[f.Len()-1]
	*f = (*f)[:f.Len()-1]
	return x
}

func topKFrequent(words []string, k int) []string {
	hp := &FrequencyHeap{}
	count := make(map[string]int)
	for _, word := range words {
		count[word]++
	}
	for word, freq := range count {
		heap.Push(hp, Frequency{word: word, freq: freq})
	}
	ans := make([]string, 0, k)
	for i := 0; i < k; i++ {
		ans = append(ans, heap.Pop(hp).(Frequency).word)
	}
	return ans
}

func findRelativeRanks(score []int) []string {
	loc := make(map[int]int)
	for i, s := range score {
		loc[s] = i
	}
	slices.Sort(score)
	slices.Reverse(score)
	ans := make([]string, len(score))
	for i, s := range score {
		if i == 0 {
			ans[loc[s]] = "Gold Medal"
		} else if i == 1 {
			ans[loc[s]] = "Silver Medal"
		} else if i == 2 {
			ans[loc[s]] = "Bronze Medal"
		} else {
			ans[loc[s]] = strconv.Itoa(i + 1)
		}
	}
	return ans
}

type KthLargest struct {
	hp IntHeap
	k  int
}

//func Constructor(k int, nums []int) KthLargest {
//	hp := &IntHeap{}
//	for _, num := range nums {
//		heap.Push(hp, num)
//		if hp.Len() > k {
//			heap.Pop(hp)
//		}
//	}
//	return KthLargest{k: k, hp: *hp}
//}
//
//func (this *KthLargest) Add(val int) int {
//	heap.Push(&this.hp, val)
//	if this.hp.Len() > this.k {
//		heap.Pop(&this.hp)
//	}
//	return this.hp[0]
//}

type IndexHeap [][2]int

func (h IndexHeap) Len() int {
	return len(h)
}

func (h IndexHeap) Less(i, j int) bool {
	return h[i][0] < h[j][0]
}

func (h IndexHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *IndexHeap) Push(x any) {
	*h = append(*h, [2]int{x.([2]int)[0], x.([2]int)[1]})
}

func (h *IndexHeap) Pop() any {
	x := (*h)[h.Len()-1]
	*h = (*h)[:h.Len()-1]
	return x
}

func maxSubsequence(nums []int, k int) (ans []int) {
	numsWithIndex := make([][2]int, len(nums))
	for i, num := range nums {
		numsWithIndex[i] = [2]int{i, num}
	}
	slices.SortFunc(numsWithIndex, func(a, b [2]int) int { return b[1] - a[1] })
	slices.SortFunc(numsWithIndex[:k], func(a, b [2]int) int { return a[0] - b[0] })
	for i := 0; i < k; i++ {
		ans = append(ans, numsWithIndex[i][1])
	}
	return
}
func largestInteger(num int) int {
	numString := strconv.Itoa(num)
	evenHeap := &IntHeap{}
	oddHeap := &IntHeap{}
	isEven := make([]bool, len(numString))
	for i, char := range numString {
		if (char-'0')%2 == 0 {
			isEven[i] = true
			heap.Push(evenHeap, int(char-'0'))
		} else {
			heap.Push(oddHeap, int(char-'0'))
		}
	}
	builder := strings.Builder{}
	for _, even := range isEven {
		if even {
			builder.WriteByte(byte(heap.Pop(evenHeap).(int) + '0'))
		} else {
			builder.WriteByte(byte(heap.Pop(oddHeap).(int) + '0'))
		}
	}
	ans, _ := strconv.Atoi(builder.String())
	return ans
}
func fillCups(amount []int) int {
	ans := 0
	slices.Sort(amount)
	for amount[2] > 0 {
		if amount[1] > 0 {
			ans++
			amount[1]--
			amount[2]--
		} else {
			ans += amount[2]
			break
		}
		slices.Sort(amount)
	}
	return ans
}
func numOfBurgers(tomatoSlices int, cheeseSlices int) []int {
	var x, y int
	m := tomatoSlices - 2*cheeseSlices
	n := 4*cheeseSlices - tomatoSlices
	if (m >= 0 && m%2 == 0) && (n >= 0 && n%2 == 0) {
		x = m / 2
		y = n / 2
		return []int{x, y}
	} else {
		return []int{}
	}
}

// type PairHeap [][]int
//
//	func (p PairHeap) Len() int {
//		return len(p)
//	}
//
//	func (p PairHeap) Less(i, j int) bool {
//		return p[i][0]+p[i][1] > p[j][0]+p[j][1]
//	}
//
//	func (p PairHeap) Swap(i, j int) {
//		p[i], p[j] = p[j], p[i]
//	}
//
//	func (p *PairHeap) Push(x any) {
//		*p = append(*p, []int{x.([]int)[0], x.([]int)[1]})
//	}
//
//	func (p *PairHeap) Pop() any {
//		x := (*p)[p.Len()-1]
//		*p = (*p)[:p.Len()-1]
//		return x
//	}
//
//	func kSmallestPairs(nums1 []int, nums2 []int, k int) [][]int {
//		hp := &PairHeap{}
//		for i := 0; i < len(nums1) && i < k; i++ {
//			for j := 0; j < len(nums2) && j < k; j++ {
//				if hp.Len() < k || nums1[i]+nums2[j] < (*hp)[0][0]+(*hp)[0][1] {
//					heap.Push(hp, []int{nums1[i], nums2[j]})
//				}
//				if hp.Len() > k {
//					heap.Pop(hp)
//				}
//			}
//		}
//		return *hp
//	}
//func kthSmallest(matrix [][]int, k int) int {
//	hp := &IntHeap{}
//	for _, row := range matrix {
//		for _, num := range row {
//			heap.Push(hp, num)
//			if hp.Len() > k {
//				heap.Pop(hp)
//			}
//		}
//	}
//	return heap.Pop(hp).(int)
//}

//func findClosestElements(arr []int, k int, x int) []int {
//	slices.SortFunc(arr, func(a, b int) int {
//		if Abs(a-x) == Abs(b-x) {
//			return a - b
//		} else {
//			return Abs(a-x) - Abs(b-x)
//		}
//	})
//	slices.Sort(arr[:k])
//	return arr[:k]
//}

// 二维数据建堆过程时间复杂度太高 可以优化
//func kthSmallestPrimeFraction(arr []int, k int) []int {
//	hp := &PairHeap{}
//	for i := 0; i < len(arr); i++ {
//		for j := i + 1; j < len(arr); j++ {
//			heap.Push(hp, []int{arr[i], arr[j]})
//			if hp.Len() > k {
//				heap.Pop(hp)
//			}
//		}
//	}
//	return heap.Pop(hp).([]int)
//}

type Suggest struct {
	productName  string
	commonPrefix int
}
type SuggestHeap []Suggest

func (s SuggestHeap) Len() int {
	return len(s)
}

func (s SuggestHeap) Less(i, j int) bool {
	if s[i].commonPrefix == s[j].commonPrefix {
		return s[i].productName < s[j].productName
	} else {
		return s[i].commonPrefix < s[j].commonPrefix
	}
}

func (s SuggestHeap) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s *SuggestHeap) Push(x any) {
	*s = append(*s, Suggest{productName: x.(Suggest).productName, commonPrefix: x.(Suggest).commonPrefix})
}

func (s *SuggestHeap) Pop() any {
	x := (*s)[s.Len()-1]
	*s = (*s)[:s.Len()-1]
	return x
}

func suggestedProducts(products []string, searchWord string) (ans [][]string) {
	commonPrefix := make([]Suggest, len(products))
	for p, product := range products {
		prefix := 0
		for i := 0; i < len(product) && i < len(searchWord); i++ {
			if product[i] == searchWord[i] {
				prefix++
			} else {
				break
			}
		}
		commonPrefix[p] = Suggest{productName: product, commonPrefix: prefix}
	}
	slices.SortFunc(commonPrefix, func(a, b Suggest) int {
		return a.commonPrefix - b.commonPrefix
	})
	for i := 1; i <= len(searchWord); i++ {
		start := sort.Search(len(commonPrefix), func(j int) bool {
			return commonPrefix[j].commonPrefix >= i
		})
		var res []string
		for j := start; j < len(commonPrefix); j++ {
			res = append(res, commonPrefix[j].productName)
		}
		slices.SortFunc(res, func(a, b string) int {
			return strings.Compare(a, b)
		})
		if len(res) >= 3 {
			ans = append(ans, res[:3])
		} else {
			ans = append(ans, res)
		}
	}
	return
}

//	type NumArray struct {
//		prefixSum []int
//		curSum    []int
//	}
//
//	func Constructor(nums []int) NumArray {
//		curSum := make([]int, len(nums))
//		prefixSum := make([]int, len(nums))
//		sum := 0
//		for i, num := range nums {
//			prefixSum[i] = sum
//			sum += num
//			curSum[i] = sum
//		}
//		return NumArray{curSum: curSum, prefixSum: prefixSum}
//	}
//
//	func (this *NumArray) SumRange(left int, right int) int {
//		return this.curSum[right] - this.prefixSum[left]
//	}
//
//	func mergeKLists(lists []*ListNode) *ListNode {
//		hp := &IntHeap{}
//		for _, list := range lists {
//			for list != nil {
//				heap.Push(hp, list.Val)
//				list = list.Next
//			}
//		}
//		if hp.Len() == 0 {
//			return nil
//		} else {
//			var head *ListNode
//			for _, list := range lists {
//				if list != nil {
//					head = lists[0]
//				}
//			}
//			p := head
//			for hp.Len() > 0 {
//				if p.Next == nil {
//					p.Next = &ListNode{Val: 0, Next: nil}
//				}
//				p = p.Next
//				p.Val = heap.Pop(hp).(int)
//			}
//			p.Next = nil
//			return head.Next
//		}
//	}
//func mergeInBetween(list1 *ListNode, a int, b int, list2 *ListNode) *ListNode {
//	left := list1
//	right := list1
//	for i := 1; i < a; i++ {
//		left = left.Next
//	}
//	for i := 0; i <= b; i++ {
//		right = right.Next
//	}
//	end := list2
//	for end.Next != nil {
//		end = end.Next
//	}
//	end.Next = right
//	left.Next = list2
//	return list1
//}

//type PairHeap [][]int
//
//func (p PairHeap) Len() int {
//	return len(p)
//}
//
//func (p PairHeap) Less(i, j int) bool {
//	return p[i][0]+p[i][1] > p[j][0]+p[j][1]
//}
//
//func (p PairHeap) Swap(i, j int) {
//	p[i], p[j] = p[j], p[i]
//}
//
//func (p *PairHeap) Push(x any) {
//	*p = append(*p, []int{x.([]int)[0], x.([]int)[1]})
//}
//
//func (p *PairHeap) Pop() any {
//	x := (*p)[p.Len()-1]
//	*p = (*p)[:p.Len()-1]
//	return x
//}

//type ListNodeHeap []*ListNode
//
//func (l ListNodeHeap) Len() int {
//	return len(l)
//}
//
//func (l ListNodeHeap) Less(i, j int) bool {
//	return l[i].Val < l[j].Val
//}
//
//func (l ListNodeHeap) Swap(i, j int) {
//	l[i], l[j] = l[j], l[i]
//}
//
//func (l *ListNodeHeap) Push(x any) {
//	*l = append(*l, &ListNode{Val: x.(*ListNode).Val, Next: x.(*ListNode).Next})
//}
//
//func (l *ListNodeHeap) Pop() any {
//	x := (*l)[l.Len()-1]
//	*l = (*l)[:l.Len()-1]
//	return x
//}

//	func mergeKLists(lists []*ListNode) *ListNode {
//		hp := &ListNodeHeap{}
//		for _, list := range lists {
//			if list != nil {
//				heap.Push(hp, list)
//			}
//		}
//		var h *ListNode
//		if hp.Len() == 0 {
//			return h
//		} else {
//			h = &ListNode{Val: -1, Next: nil}
//			p := h
//			for hp.Len() > 0 {
//				p.Next = &ListNode{Val: -1, Next: nil}
//				p = p.Next
//				node := heap.Pop(hp).(*ListNode)
//				p.Val = node.Val
//				if node.Next != nil {
//					heap.Push(hp, node.Next)
//				}
//			}
//			return h.Next
//		}
//	}
func similarPairs(words []string) int {
	hash := make(map[[26]bool]int, len(words))
	for _, word := range words {
		existence := [26]bool{}
		for _, char := range word {
			existence[char-'a'] = true
		}
		hash[existence]++
	}
	ans := 0
	for _, v := range hash {
		ans += v * (v - 1) / 2
	}
	return ans
}
func tictactoe(moves [][]int) string {
	row := make([]int, 3)
	col := make([]int, 3)
	diag := make([]int, 2)
	for i := 0; i < len(moves); i++ {
		move := moves[i]
		if i%2 == 0 {
			row[move[0]]++
			col[move[1]]++
			if move[0] == move[1] {
				diag[0]++
			}
			if move[0]+move[1] == 2 {
				diag[1]++
			}
		} else {
			row[move[0]]--
			col[move[1]]--
			if move[0] == move[1] {
				diag[0]--
			}
			if move[0]+move[1] == 2 {
				diag[1]--
			}
		}
	}
	for i := 0; i < 3; i++ {
		if row[i] == 3 {
			return "A"
		} else if row[i] == -3 {
			return "B"
		}
	}
	for i := 0; i < 3; i++ {
		if col[i] == 3 {
			return "A"
		} else if col[i] == -3 {
			return "B"
		}
	}
	for i := 0; i < 2; i++ {
		if diag[i] == 3 {
			return "A"
		} else if diag[i] == -3 {
			return "B"
		}
	}
	if len(moves) < 9 {
		return "Pending"
	}
	return "Draw"
}
func shiftGrid(grid [][]int, k int) (ans [][]int) {
	arr := make([]int, 0, len(grid)*len(grid[0]))
	for _, row := range grid {
		for _, num := range row {
			arr = append(arr, num)
		}
	}
	newGrid := append(arr[len(arr)-(k%len(arr)):], arr[:len(arr)-(k%len(arr))]...)
	for i := 0; i < len(grid); i++ {
		res := make([]int, len(grid[0]))
		for j := 0; j < len(grid[0]); j++ {
			res[j] = newGrid[i*len(grid[0])+j]
		}
		ans = append(ans, res)
	}
	return
}
func pivotArray(nums []int, pivot int) []int {
	slices.SortStableFunc(nums, func(a, b int) int {
		if (a-pivot)*(b-pivot) < 0 {
			return a - b
		} else if a == pivot || b == pivot {
			return a - b
		} else {
			return 0
		}
	})
	return nums
}

//	type RecentCounter struct {
//		pingSlice []int
//	}
//
//	func Constructor() RecentCounter {
//		return RecentCounter{pingSlice: make([]int, 0)}
//	}
//
//	func (this *RecentCounter) Ping(t int) int {
//		this.pingSlice = append(this.pingSlice, t)
//		return len(this.pingSlice) - sort.SearchInts(this.pingSlice, t-3000)
//	}
func sortColors(nums []int) {
	index := 0
	for i := 0; i < len(nums); i++ {
		if nums[index] == 0 {
			for j := index - 1; j >= 0; j-- {
				nums[j+1] = nums[j]
			}
			nums[0] = 0
			index++
		} else if nums[index] == 2 {
			nums = append(nums[:index], append(nums[index+1:], 2)...)
		} else {
			index++
		}
	}
}
func partitionLabels(s string) (ans []int) {
	for i := 0; i < len(s); {
		lastLoc, j := i, i
		for ; j <= lastLoc; j++ {
			lastLoc = max(lastLoc, strings.LastIndexByte(s, s[j]))
		}
		ans = append(ans, j-i)
		i = j
	}
	return
}
func decodeString(s string) string {
	numberStack := make([]int, 0, len(s))
	stringStack := make([]string, 0, len(s))
	numberBuffer := strings.Builder{}
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			numberBuffer.WriteByte(s[i])
		} else if s[i] == '[' {
			number, _ := strconv.Atoi(numberBuffer.String())
			numberBuffer.Reset()
			numberStack = append(numberStack, number)
			stringStack = append(stringStack, s[i:i+1])
		} else if s[i] >= 'a' && s[i] <= 'z' {
			stringStack = append(stringStack, s[i:i+1])
		} else {
			start := len(stringStack) - 1
			for start >= 0 && stringStack[start] != "[" {
				start--
			}
			times := numberStack[len(numberStack)-1]
			numberStack = numberStack[:len(numberStack)-1]
			var res string
			for time := 0; time < times; time++ {
				for j := start + 1; j < len(stringStack); j++ {
					res += stringStack[j]
				}
			}
			stringStack = stringStack[:start]
			stringStack = append(stringStack, res)
		}
	}
	builder := strings.Builder{}
	for i := 0; i < len(stringStack); i++ {
		builder.WriteString(stringStack[i])
	}
	return builder.String()
}

//func searchMatrix(matrix [][]int, target int) bool {
//	var searchSlice []int
//	for _, row := range matrix {
//		if target < row[len(row)-1] {
//			searchSlice = row
//			break
//		} else if target == row[len(row)-1] {
//			return true
//		}
//	}
//	_, ok := slices.BinarySearch(searchSlice, target)
//	return ok
//}

//type Node struct {
//	Val    int
//	Next   *Node
//	Random *Node
//}

func copyRandomList(head *Node) *Node {
	dummy := &Node{}
	nodeHash := make(map[*Node]*Node)
	for p, q := head, dummy; p != nil; p = p.Next {
		q.Next = &Node{}
		q = q.Next
		q.Val = p.Val
		nodeHash[p] = q
	}
	for p, q := head, dummy.Next; p != nil && q != nil; p, q = p.Next, q.Next {
		q.Random = nodeHash[p.Random]
	}
	return dummy.Next
}

//func largestRectangleArea(heights []int) int {
//	ans := math.MinInt
//	stack := make([]int, 0, len(heights))
//	for i, height := range heights {
//		if len(stack) == 0 || height > heights[stack[len(stack)-1]] {
//			stack = append(stack, i)
//		} else {
//			for len(stack) > 0 && heights[stack[len(stack)-1]] > height {
//				if len(stack) > 1 {
//					ans = max(ans, (i-stack[len(stack)-2]-1)*heights[stack[len(stack)-1]])
//				} else {
//					ans = max(ans, i*heights[stack[len(stack)-1]])
//				}
//				stack = stack[:len(stack)-1]
//			}
//			stack = append(stack, i)
//		}
//	}
//	for len(stack) > 0 {
//		n := 0
//		if len(stack) > 1 {
//			n = len(heights) - stack[len(stack)-2] - 1
//		} else {
//			n = len(heights)
//		}
//		ans = max(ans, n*heights[stack[len(stack)-1]])
//		stack = stack[:len(stack)-1]
//	}
//	return ans
//}
