package main

import (
	"fmt"
	"slices"
	"sort"
	"strconv"
)

//func twoSum(nums []int, target int) []int {
//	hashmap := make(map[int]int)
//	for i, num := range nums {
//		if loc, ok := hashmap[target-num]; ok {
//			return []int{i, loc}
//		}
//		hashmap[num] = i
//	}
//	return nil
//}

//func reverseNum(x int) int {
//	value := 0
//	for ; x > 0; x = x / 10 {
//		value *= 10
//		value += x % 10
//	}
//	return value
//}

//	func isPalindrome(x int) bool {
//		if x == 0 {
//			return true
//		} else if x < 0 || x%10 == 0 {
//			return false
//		} else {
//			value := 0
//			for ; x > value; x = x / 10 {
//				value *= 10
//				value += x % 10
//			}
//			return x == value || x == value/10
//		}
//	}
func romanToInt(s string) int {
	romanChars := map[string]int{
		"I": 1,
		"V": 5,
		"X": 10,
		"L": 50,
		"C": 100,
		"D": 500,
		"M": 1000,
	}
	romanGroups := map[string]int{
		"IV": 4,
		"IX": 9,
		"XL": 40,
		"XC": 90,
		"CD": 400,
		"CM": 900,
	}
	res := 0
	for i := len(s) - 1; i >= 0; i-- {
		if i == 0 {
			value, _ := romanChars[string(s[i])]
			res += value
		} else {
			if value, ok := romanGroups[string(s[i-1])+string(s[i])]; ok {
				res += value
				i--
			} else {
				value, _ = romanChars[string(s[i])]
				res += value
			}
		}
	}
	return res
}

//	func longestCommonPrefix(strs []string) string {
//		prefixLen := len(strs[0])
//		for i := 0; i < len(strs)-1 && prefixLen > 0; i++ {
//			prefixLen = min(prefixLen, len(strs[i+1]))
//			for prefixLen > 0 {
//				if strs[i][:prefixLen] == strs[i+1][:prefixLen] {
//					break
//				}
//				prefixLen--
//			}
//		}
//		return strs[0][:prefixLen]
//	}
func isValid(s string) bool {
	var stack []byte
	for i, v := range s {
		if string(v) == "(" || string(v) == "[" || string(v) == "{" {
			stack = append(stack, s[i])
		} else {
			if len(stack) == 0 {
				return false
			}
			pop := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			ok := (string(v) == ")" && string(pop) == "(") || (string(v) == "]" && string(pop) == "[") || (string(v) == "}" && string(pop) == "{")
			if !ok {
				return false
			}
		}
	}
	return len(stack) == 0
}

type ListNode struct {
	Val  int
	Next *ListNode
}

//func partition(head *ListNode, x int) *ListNode {
//	var small *ListNode
//	var big *ListNode
//	var sp *ListNode
//	var bp *ListNode
//	for ; head != nil; head = head.Next {
//		if head.Val < x {
//			if small == nil {
//				small = &ListNode{Val: head.Val, Next: nil}
//				sp = small
//			} else {
//				sp.Next = &ListNode{Val: head.Val, Next: nil}
//				sp = sp.Next
//			}
//		} else {
//			if big == nil {
//				big = &ListNode{Val: head.Val, Next: nil}
//				bp = big
//			} else {
//				bp.Next = &ListNode{Val: head.Val, Next: nil}
//				bp = bp.Next
//			}
//		}
//	}
//	if sp == nil {
//		return big
//	} else {
//		sp.Next = big
//		return small
//	}
//}

//迭代法
//func reverseList(head *ListNode) *ListNode {
//	if head == nil || head.Next == nil {
//		return head
//	} else {
//		var pre *ListNode
//		var next *ListNode
//		for head != nil {
//			next = head.Next
//			head.Next = pre
//			pre = head
//			head = next
//		}
//		return pre
//	}
//}

//func isPalindrome(head *ListNode) bool {
//	if head == nil || head.Next == nil {
//		return true
//	} else {
//		slow := head
//		fast := head
//		var pre *ListNode
//		var next *ListNode
//		for fast != nil && fast.Next != nil {
//			fast = fast.Next.Next
//			next = slow.Next
//			slow.Next = pre
//			pre = slow
//			slow = next
//		}
//		if fast != nil {
//			slow = slow.Next
//		}
//		for ; pre != nil && slow != nil; pre, slow = pre.Next, slow.Next {
//			if pre.Val != slow.Val {
//				return false
//			}
//		}
//		return true
//	}
//}

//func deleteDuplicates(head *ListNode) *ListNode {
//	set := make(map[int]struct{})
//	p := head
//	prev := p
//	for p != nil {
//		if _, ok := set[p.Val]; ok {
//			prev.Next = p.Next
//			p = p.Next
//		} else {
//			set[p.Val] = struct{}{}
//			prev = p
//			p = p.Next
//		}
//	}
//	return head
//}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	} else if list2 == nil {
		return list1
	}
	if list1.Val > list2.Val {
		list1, list2 = list2, list1
	}
	p1 := list1
	p2 := list2
	for p1 != nil && p2 != nil {
		if p1.Val <= p2.Val && (p1.Next == nil || p2.Val <= p1.Next.Val) {
			temp := p2
			p2 = p2.Next
			temp.Next = p1.Next
			p1.Next = temp
			p1 = temp
		} else {
			p1 = p1.Next
		}
	}
	return list1
}

//func removeDuplicates(nums []int) int {
//	slices.Sort(nums)
//	set := make(map[int]int)
//	for i := 0; i < len(nums); i++ {
//		if _, ok := set[nums[i]]; ok {
//			nums[i] = 10001
//		} else {
//			set[nums[i]] = i
//		}
//	}
//	slices.Sort(nums)
//	return len(set)
//}

func removeElement(nums []int, val int) int {
	slow := 0
	for fast := 0; fast < len(nums); fast++ {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
		}
	}
	return slow
}

func lengthOfLastWord(s string) int {
	count := 1
	for i := len(s) - 1; i > 0; i-- {
		if string(s[i]) != " " {
			if string(s[i-1]) != " " {
				count++
			} else {
				break
			}
		}
	}
	return count
}
func plusOne(digits []int) []int {
	digits[len(digits)-1] += +1
	for i := len(digits) - 1; i > 0; i-- {
		if digits[i] == 10 {
			digits[i] = 0
			digits[i-1] += 1
		}
	}
	if digits[0] == 10 {
		digits[0] = 0
		digits = append(digits[:0], append([]int{1}, digits[0:]...)...)
	}
	return digits
}

func addBinary(a string, b string) string {
	if a == "0" {
		return b
	} else if b == "0" {
		return a
	} else {
		res := ""
		carry := 0
		i := len(a) - 1
		j := len(b) - 1
		for i >= 0 || j >= 0 {
			val1 := 0
			val2 := 0
			if i >= 0 {
				val1, _ = strconv.Atoi(string(a[i]))
			}
			if j >= 0 {
				val2, _ = strconv.Atoi(string(b[j]))
			}
			val := val1 + val2 + carry
			res = strconv.Itoa(val%2) + res
			if val == 0 || val == 1 {
				carry = 0
			} else if val == 2 || val == 3 {
				carry = 1
			}
			i--
			j--
		}
		if carry != 0 {
			res = "1" + res
		}
		return res
	}
}
func mySqrt(x int) int {
	i := 0
	for ; i < x; i++ {
		if i*i <= x && (i+1)*(i+1) > x {
			break
		}
	}
	return i
}

//func merge(nums1 []int, m int, nums2 []int, n int) {
//	slices.Sort(append(nums1[:m], nums2[:n]...))
//}

func isBadVersion(version int) bool {
	return true
}
func firstBadVersion(n int) int {
	l := 1
	for l <= n {
		if isBadVersion((l + n) / 2) {
			n = (l+n)/2 - 1
		} else {
			l = (l+n)/2 + 1
		}
	}
	return l
}

func containsDuplicate(nums []int) bool {
	set := make(map[int]struct{})
	ok := false
	for _, num := range nums {
		if _, ok = set[num]; ok {
			break
		} else {
			set[num] = struct{}{}
		}
	}
	return ok
}

func majorityElement(nums []int) int {
	set := make(map[int]int)
	num := 0
	for _, num = range nums {
		if _, ok := set[num]; ok {
			if set[num]+1 > len(nums)/2 {
				return num
			} else {
				set[num]++
			}
		} else {
			set[num] = 1
		}
	}
	return num
}

func sortedSquares(nums []int) []int {
	newNums := make([]int, len(nums))
	l := 0
	r := len(nums) - 1
	for l <= r {
		leftVal := nums[l] * nums[l]
		rightVal := nums[r] * nums[r]
		if leftVal >= rightVal {
			newNums[r-l] = leftVal
			l++
		} else {
			newNums[r-l] = rightVal
			r--
		}
	}
	return newNums
}
func findPeakElement(nums []int) int {
	if len(nums) == 1 || nums[0] > nums[1] {
		return 0
	} else if nums[len(nums)-1] > nums[len(nums)-2] {
		return len(nums) - 1
	} else {
		l := 1
		r := len(nums) - 2
		m := 0
		for l <= r {
			m = l + (r-l)/2
			if nums[m] > nums[m-1] && nums[m] > nums[m+1] {
				break
			} else if nums[m] > nums[m-1] {
				l = m + 1
			} else {
				r = m - 1
			}
		}
		return m
	}
}
func peakIndexInMountainArray(arr []int) int {
	l := 0
	r := len(arr) - 1
	m := 0
	for l <= r {
		m = l + (r-l)/2
		if (m == 0 || arr[m-1] < arr[m]) && (m == len(arr)-1 || arr[m] > arr[m+1]) {
			break
		} else if m == 0 || arr[m] > arr[m-1] {
			l = m + 1
		} else {
			r = m - 1
		}
	}
	return m
}

//func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
//	carry := 0
//	val2 := 0
//	head := l1
//	for ; l1 != nil; l1 = l1.Next {
//		if l2 != nil {
//			val2 = l2.Val
//			l2 = l2.Next
//		} else {
//			val2 = 0
//		}
//		val := l1.Val + val2 + carry
//		l1.Val = val % 10
//		carry = val / 10
//		if l1.Next == nil && (l2 != nil || carry != 0) {
//			l1.Next = &ListNode{}
//		}
//	}
//	return head
//}

//	func swapPairs(head *ListNode) *ListNode {
//		if head == nil || head.Next == nil {
//			return head
//		} else {
//			cur := head
//			head = cur.Next
//			var pre *ListNode
//			for cur != nil {
//				next := cur.Next
//				if next != nil {
//					cur.Next = next.Next
//					next.Next = cur
//					if pre != nil {
//						pre.Next = next
//					}
//					pre = cur
//				}
//				cur = cur.Next
//			}
//			return head
//		}
//	}
func swap(pre *ListNode, cur *ListNode) {
	if cur != nil {
		next := cur.Next
		if next != nil {
			cur.Next = next.Next
			next.Next = cur
			if pre != nil {
				pre.Next = next
			}
			pre = cur
		}
		swap(pre, cur.Next)
	}
}
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	} else {
		cur := head
		head = cur.Next
		var pre *ListNode
		swap(pre, cur)
		return head
	}
}

//func removeNthFromEnd(head *ListNode, n int) *ListNode {
//	head = &ListNode{Val: -1, Next: head}
//	slow, fast := head, head
//	for i := 0; i < n; i++ {
//		fast = fast.Next
//	}
//	for fast.Next != nil {
//		fast, slow = fast.Next, slow.Next
//	}
//	slow.Next = slow.Next.Next
//	return head.Next
//}

//func deleteNode(node *ListNode) {
//	node.Val = node.Next.Val
//	node.Next = node.Next.Next
//}

type MinStack struct {
	stack []int
	min   []int
}

//func Constructor() MinStack {
//	return MinStack{stack: make([]int, 0), min: make([]int, 0)}
//}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	if len(this.min) == 0 {
		this.min = append(this.min, val)
	} else {
		this.min = append(this.min, min(this.min[len(this.min)-1], val))
	}
}

func (this *MinStack) Pop() {
	if len(this.stack) > 0 {
		this.stack = this.stack[:len(this.stack)-1]
		this.min = this.min[:len(this.min)-1]
	}
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.min[len(this.min)-1]
}

type MyCircularDeque struct {
	queue []int
	size  int
}

//func Constructor(k int) MyCircularDeque {
//	return MyCircularDeque{queue: make([]int, 0), size: k}
//}

func (this *MyCircularDeque) InsertFront(value int) bool {
	if this.IsFull() {
		return false
	} else {
		this.queue = append([]int{value}, this.queue...)
		return true
	}
}

func (this *MyCircularDeque) InsertLast(value int) bool {
	if this.IsFull() {
		return false
	} else {
		this.queue = append(this.queue, []int{value}...)
		return true
	}
}

func (this *MyCircularDeque) DeleteFront() bool {
	if !this.IsEmpty() {
		this.queue = this.queue[1:]
		return true
	} else {
		return false
	}
}

func (this *MyCircularDeque) DeleteLast() bool {
	if !this.IsEmpty() {
		this.queue = this.queue[:len(this.queue)-1]
		return true
	} else {
		return false
	}
}

func (this *MyCircularDeque) GetFront() int {
	if this.IsEmpty() {
		return -1
	} else {
		return this.queue[0]
	}
}

func (this *MyCircularDeque) GetRear() int {
	if this.IsEmpty() {
		return -1
	} else {
		return this.queue[len(this.queue)-1]
	}
}

func (this *MyCircularDeque) IsEmpty() bool {
	return len(this.queue) == 0
}

func (this *MyCircularDeque) IsFull() bool {
	return len(this.queue) == this.size
}
func successfulPairs(spells []int, potions []int, success int64) (pairCount []int) {
	slices.Sort(potions)
	length := len(potions)
	for _, spell := range spells {
		pairCount = append(pairCount, length-sort.Search(length, func(i int) bool {
			return int64(potions[i]*spell) >= success
		}))
	}
	return
}

//func preorderTraversal(root *TreeNode) (order []int) {
//	if root != nil {
//		stack := make([]*TreeNode, 0)
//		stack = append(stack, root)
//		for len(stack) > 0 {
//			order = append(order, stack[len(stack)-1].Val)
//			root, stack = stack[len(stack)-1], stack[:len(stack)-1]
//			if root.Right != nil {
//				stack = append(stack, root.Right)
//			}
//			if root.Left != nil {
//				stack = append(stack, root.Left)
//			}
//		}
//	}
//	return
//}

func middleNode(head *ListNode) *ListNode {
	for fast := head; fast != nil && fast.Next != nil; {
		head = head.Next
		fast = fast.Next.Next
	}
	return head
}

//	func hasCycle(head *ListNode) bool {
//		for fast := head; fast != nil && fast.Next != nil; {
//			head = head.Next
//			fast = fast.Next.Next
//			if head == fast {
//				return true
//			}
//		}
//		return false
//	}
func detectCycle(head *ListNode) *ListNode {
	slow := head
	fast := head
	hasCycle := false
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			hasCycle = true
			break
		}
	}
	if hasCycle {
		for head != slow {
			head = head.Next
			slow = slow.Next
		}
		return head
	} else {
		return nil
	}
}

//	func reverseList(head *ListNode) *ListNode {
//		if head == nil || head.Next == nil {
//			return head
//		} else {
//			var pre *ListNode
//			var next *ListNode
//			for head != nil {
//				next = head.Next
//				head.Next = pre
//				pre = head
//				head = next
//			}
//			return pre
//		}
//	}
//func reorderList(head *ListNode) {
//	middle := middleNode(head)
//	middle.Next = reverseList(middle.Next)
//	p := head
//	for middle.Next != nil {
//		next := middle.Next
//		middle.Next = next.Next
//		next.Next = p.Next
//		p.Next = next
//		p = next.Next
//	}
//}

//	func isAnagram(s string, t string) bool {
//		if len(s) != len(t) {
//			return false
//		} else {
//			charCount1 := make(map[string]int)
//			charCount2 := make(map[string]int)
//			for i := 0; i < len(s); i++ {
//				if count, ok := charCount1[s[i:i+1]]; ok {
//					charCount1[s[i:i+1]] = count + 1
//				} else {
//					charCount1[s[i:i+1]] = 1
//				}
//				if count, ok := charCount2[t[i:i+1]]; ok {
//					charCount2[t[i:i+1]] = count + 1
//				} else {
//					charCount2[t[i:i+1]] = 1
//				}
//			}
//			for k := range charCount1 {
//				if _, ok := charCount2[k]; ok {
//					if charCount1[k] != charCount2[k] {
//						return false
//					}
//				} else {
//					return false
//				}
//			}
//			return true
//		}
//	}
//
//	func isAnagram(s string, t string) bool {
//		if len(s) != len(t) {
//			return false
//		} else {
//			charCount := make(map[string]*[2]int)
//			for i := 0; i < len(s); i++ {
//				if count, ok := charCount[s[i:i+1]]; ok {
//					count[0] = count[0] + 1
//				} else {
//					charCount[s[i:i+1]] = &[2]int{1, 0}
//				}
//				if count, ok := charCount[t[i:i+1]]; ok {
//					count[1] = count[1] + 1
//				} else {
//					charCount[t[i:i+1]] = &[2]int{0, 1}
//				}
//			}
//			for _, v := range charCount {
//				if v[0] != v[1] {
//					return false
//				}
//			}
//			return true
//		}
//	}
//
//	func intersection(nums1 []int, nums2 []int) (ans []int) {
//		set := make(map[int]struct{})
//		for _, num := range nums1 {
//			if _, ok := set[num]; !ok {
//				set[num] = struct{}{}
//			}
//		}
//		for _, num := range nums2 {
//			if _, ok := set[num]; ok {
//				ans = append(ans, num)
//				delete(set, num)
//			}
//		}
//		return
//	}
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) (ans int) {
	sum1 := make(map[int]int, len(nums1)*len(nums1))
	sum2 := make(map[int]int, len(nums1)*len(nums1))
	for i := 0; i < len(nums1); i++ {
		for j := 0; j < len(nums1); j++ {
			if _, ok := sum1[nums1[i]+nums2[j]]; ok {
				sum1[nums1[i]+nums2[j]]++
			} else {
				sum1[nums1[i]+nums2[j]] = 1
			}
			if _, ok := sum2[nums3[i]+nums4[j]]; ok {
				sum2[nums3[i]+nums4[j]]++
			} else {
				sum2[nums3[i]+nums4[j]] = 1
			}
		}
	}
	for k, v := range sum1 {
		if count, ok := sum2[-k]; ok {
			ans += count * v
		}
	}
	return
}

//type ThreeTuple struct {
//	left   int
//	middle int
//	right  int
//}

//func threeSum(nums []int) (ans [][]int) {
//	slices.Sort(nums)
//	for i, n1 := range nums {
//		if n1 <= 0 {
//			if i > 0 && n1 == nums[i-1] {
//				continue
//			}
//			left := i + 1
//			right := len(nums) - 1
//			for left < right {
//				n2, n3 := nums[left], nums[right]
//				if n1+n2+n3 > 0 {
//					right--
//				} else if n1+n2+n3 < 0 {
//					left++
//				} else {
//					ans = append(ans, []int{n1, n2, n3})
//					for right > left && n2 == nums[left] {
//						left++
//					}
//					for right > left && n2 == nums[right] {
//						right--
//					}
//				}
//			}
//		}
//	}
//	return
//}

//	func searchAndAdd(set map[ThreeTuple]struct{}, p int, q int, nums []int) {
//		left := nums[p]
//		right := nums[q]
//		target := -(left + right)
//		searchSlice := nums[p+1 : q]
//		targetIndex := search(searchSlice, target)
//
//		if targetIndex != -1 {
//			threeTuple := ThreeTuple{left: left, middle: target, right: right}
//			if _, ok := set[threeTuple]; !ok {
//				set[threeTuple] = struct{}{}
//			}
//		}
//	}
//
//	func threeSum(nums []int) (ans [][]int) {
//		slices.Sort(nums)
//		set := make(map[ThreeTuple]struct{})
//		p := 0
//		q := len(nums) - 1
//		for q-p >= 2 {
//			qq := q
//			for qq-p >= 2 {
//				if !(nums[qq]+nums[qq-1]+nums[p] < 0) {
//					searchAndAdd(set, p, qq, nums)
//					qq--
//				} else {
//					break
//				}
//			}
//			pp := p
//			for q-pp >= 2 {
//				if !(nums[pp]+nums[pp+1]+nums[q] > 0) {
//					searchAndAdd(set, pp, q, nums)
//					pp++
//				} else {
//					break
//				}
//			}
//			p++
//			q--
//		}
//		for k, _ := range set {
//			ans = append(ans, []int{k.left, k.middle, k.right})
//		}
//		return
//	}
func twoSum(numbers []int, target int) []int {
	left := 0
	right := len(numbers) - 1
	for left < right {
		if numbers[left]+numbers[right] == target {
			return []int{left + 1, right + 1}
		} else if numbers[left]+numbers[right] < target {
			left++
		} else {
			right--
		}
	}
	return []int{}
}
func fourSum(nums []int, target int) (ans [][]int) {
	slices.Sort(nums)
	length := len(nums)
	for i := 0; i <= length-4; i++ {
		n1 := nums[i]
		if i > 0 && n1 == nums[i-1] {
			continue
		}
		if n1+nums[i+1]+nums[i+2]+nums[i+3] > target {
			break
		}
		if n1+nums[length-1]+nums[length-2]+nums[length-3] < target {
			break
		}
		for j := i + 1; j <= length-3; j++ {
			n2 := nums[j]
			if j > i+1 && n2 == nums[j-1] {
				continue
			}
			left := j + 1
			right := length - 1
			for left < right {
				n3 := nums[left]
				n4 := nums[right]
				if n1+n2 == -(n3 + n4 - target) {
					ans = append(ans, []int{n1, n2, n3, n4})
					for left < right && n3 == nums[left] {
						left++
					}
					for left < right && n4 == nums[right] {
						right--
					}
				} else if n1+n2 > -(n3 + n4 - target) {
					right--
				} else {
					left++
				}
			}
		}
	}
	return
}
func main() {
	fmt.Println(reverseNum(300))
}
