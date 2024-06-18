package main

import (
	"container/heap"
	"container/list"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
)

func getIntersectionNode__(headA, headB *ListNode) *ListNode {
	mp := make(map[*ListNode]struct{})
	for p := headA; p != nil; {
		mp[p] = struct{}{}
		p = p.Next
	}
	for p := headB; p != nil; {
		if _, ok := mp[p]; ok {
			return p
		}
		p = p.Next
	}
	return nil
}

//func reverseList__(head *ListNode) *ListNode {
//	var pre *ListNode
//	for cur := head; cur != nil; {
//		nxt := cur.Next
//		cur.Next = pre
//		pre, cur = cur, nxt
//	}
//	return pre
//}

// 先找到链表的中点 反转
//
//	func isPalindrome__(head *ListNode) bool {
//		slow, fast := head, head
//		for fast != nil && fast.Next != nil {
//			slow = slow.Next
//			fast = fast.Next.Next
//		}
//		reversedHead := reverseList(slow)
//		for head != nil && reversedHead != nil {
//			if head.Val != reversedHead.Val {
//				return false
//			}
//			head = head.Next
//			reversedHead = reversedHead.Next
//		}
//		return true
//	}
func hasCycle__(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

func detectCycle__(head *ListNode) *ListNode {
	slow, fast := head, head
	cycle := false
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			cycle = true
			break
		}
	}
	if !cycle {
		return nil
	}
	for p := head; p != slow; {
		p = p.Next
		slow = slow.Next
	}
	return slow
}
func mergeTwoLists_(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	p1, p2, p3 := list1, list2, dummy
	for p1 != nil && p2 != nil {
		if p1.Val <= p2.Val {
			p3.Next = &ListNode{Val: p1.Val}
			p3 = p3.Next
			p1 = p1.Next
		} else {
			p3.Next = &ListNode{Val: p2.Val}
			p3 = p3.Next
			p2 = p2.Next
		}
	}
	if p1 != nil {
		p3.Next = p1
	}
	if p2 != nil {
		p3.Next = p2
	}
	return dummy.Next
}
func addTwoNumbers_(l1 *ListNode, l2 *ListNode) *ListNode {
	p1, p2, dummy := l1, l2, &ListNode{}
	p3 := dummy
	carry := 0
	for p1 != nil && p2 != nil {
		val := (carry + p1.Val + p2.Val) % 10
		carry = (carry + p1.Val + p2.Val) / 10
		p3.Next = &ListNode{Val: val}
		p1, p2, p3 = p1.Next, p2.Next, p3.Next
	}
	var p *ListNode
	if p1 != nil {
		p = p1
	} else {
		p = p2
	}
	for p != nil {
		val := (carry + p.Val) % 10
		carry = (carry + p.Val) / 10
		p3.Next = &ListNode{Val: val}
		p, p3 = p.Next, p3.Next
	}
	if carry != 0 {
		p3.Next = &ListNode{Val: 1}
	}
	return dummy.Next
}
func removeNthFromEnd_(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	slow, fast := dummy, dummy
	for ; n > 0; n-- {
		fast = fast.Next
	}
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

// 一维差分
//func maximumBeauty(nums []int, k int) int {
//	left := math.MaxInt
//	right := math.MinInt
//	beauty := make([][]int, 0, len(nums))
//	for _, num := range nums {
//		beauty = append(beauty, []int{num - k, num + k})
//		left = min(left, num-k)
//		right = max(right, num+k)
//	}
//	diff := make([]int, right-left+2)
//	for _, b := range beauty {
//		diff[b[0]-left]++
//		diff[b[1]-left+1]--
//	}
//	prefix := make([]int, len(diff))
//	prefix[0] = diff[0]
//	ans := prefix[0]
//	for i := 1; i < len(prefix); i++ {
//		prefix[i] = prefix[i-1] + diff[i]
//		ans = max(prefix[i], ans)
//	}
//	return ans
//}

func maximumBeauty(nums []int, k int) int {
	slices.Sort(nums)
	var left, ans int
	for right := 0; right < len(nums); right++ {
		for nums[right]-nums[left] > k*2 {
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}
func swapPairs_(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	for pre, cur := dummy, dummy.Next; cur != nil && cur.Next != nil; {
		nxt := cur.Next
		pre.Next = nxt
		cur.Next = nxt.Next
		nxt.Next = cur
		pre = cur
		cur = cur.Next
	}
	return dummy.Next
}
func reverseKGroup(head *ListNode, k int) *ListNode {
	n := 0
	for p := head; p != nil; p = p.Next {
		n++
	}
	dummy := &ListNode{Next: head}
	temp := dummy
	cur := dummy.Next
	var pre *ListNode
	for i := 0; i < n/k; i++ {
		for j := 0; j < k; j++ {
			nxt := cur.Next
			cur.Next = pre
			pre, cur = cur, nxt
		}
		temp.Next.Next = cur
		nextTemp := temp.Next
		temp.Next = pre
		temp = nextTemp
	}
	return dummy.Next
}
func reverseBetween_(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	temp := dummy
	for i := 1; i < left; i++ {
		temp = temp.Next
	}
	cur := temp.Next
	var pre *ListNode
	for i := 0; i < right-left+1; i++ {
		nxt := cur.Next
		cur.Next = pre
		pre, cur = cur, nxt
	}
	temp.Next.Next = cur
	temp.Next = pre
	return dummy.Next
}
func sortList(head *ListNode) *ListNode {
	list := make([]*ListNode, 0, 1000)
	for p := head; p != nil; p = p.Next {
		list = append(list, p)
	}
	slices.SortFunc(list, func(a, b *ListNode) int {
		return a.Val - b.Val
	})
	dummy := &ListNode{}
	p := dummy
	for _, node := range list {
		p.Next = node
		p = p.Next
	}
	return dummy.Next
}
func insert(intervals [][]int, newInterval []int) [][]int {
	start := 0
	for ; start < len(intervals); start++ {
		if !(intervals[start][0] > newInterval[1] || intervals[start][1] < newInterval[0]) {
			break
		}
	}
	if start == len(intervals) {
		index := sort.Search(len(intervals), func(i int) bool {
			return intervals[i][0] >= newInterval[0]
		})
		intervals = append(intervals[:index], append([][]int{newInterval}, intervals[index:]...)...)
		return intervals
	}
	ans := make([][]int, 0, len(intervals))
	ans = append(ans, intervals[:start]...)
	pre := []int{min(intervals[start][0], newInterval[0]), max(intervals[start][1], newInterval[1])}
	i := start + 1
	for ; i < len(intervals); i++ {
		if intervals[i][0] > pre[1] || intervals[i][1] < pre[0] {
			break
		} else {
			pre[0], pre[1] = min(pre[0], intervals[i][0]), max(pre[1], intervals[i][1])
		}
	}
	ans = append(ans, pre)
	ans = append(ans, intervals[i:]...)
	return ans
}
func getListLen(node *ListNode) (ans int) {
	for p := node; p != nil; {
		ans++
		p = p.Next
	}
	return ans
}
func rotateRight(head *ListNode, k int) *ListNode {
	//先求出整个链表的长度
	n := getListLen(head)
	if n == 0 || k%n == 0 {
		return head
	}
	k %= n
	//需要找到倒数第k个节点:也就是正数第n-k+1个节点和正数第n-k个节点
	p1 := head
	for i := 0; i < n-k-1; i++ {
		p1 = p1.Next
	}
	p2 := p1.Next
	p1.Next = nil
	p3 := p2
	for p3.Next != nil {
		p3 = p3.Next
	}
	p3.Next = head
	return p2
}

// 返回第一个>=target的index
func lowerBound(nums []int, target int) int {
	left, right := 0, len(nums)-1
	ans := len(nums)
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] >= target {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return ans
}
func searchRange(nums []int, target int) []int {
	index := lowerBound(nums, target)
	if index == len(nums) || nums[index] != target {
		return []int{-1, -1}
	} else {
		return []int{index, lowerBound(nums, target+1) - 1}
	}
}
func searchInsert(nums []int, target int) int {
	return lowerBound(nums, target)
}
func search___(nums []int, target int) int {
	index := lowerBound(nums, target)
	if index == len(nums) || nums[index] != target {
		return -1
	}
	return index
}
func nextGreatestLetter(letters []byte, target byte) byte {
	left, right := 0, len(letters)-1
	ans := 0
	for left <= right {
		mid := (right-left)/2 + left
		if letters[mid] > target {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return letters[ans]
}

func maximumCount_(nums []int) int {
	positive := len(nums) - lowerBound(nums, 1)
	negative := lowerBound(nums, 0)
	return max(positive, negative)
}

func findLUSlength(a string, b string) int {
	if a == b {
		return -1
	}
	return max(len(a), len(b))
}
func successfulPairs(spells []int, potions []int, success int64) []int {
	ans := make([]int, len(spells))
	slices.Sort(potions)
	for index, spell := range spells {
		ans[index] = len(potions) - sort.Search(len(potions), func(i int) bool {
			return int64(spell)*int64(potions[i]) >= success
		})
	}
	return ans
}
func answerQueries_(nums []int, queries []int) []int {
	slices.Sort(nums)
	ans := make([]int, len(queries))
	prefix := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] + nums[i]
	}
	for i := 0; i < len(ans); i++ {
		ans[i] = sort.Search(len(nums), func(j int) bool {
			return prefix[j+1] > queries[i] //第一个大于的
		})
	}
	return ans
}

type LRUCache struct {
	//放在最前面的意味着最近访问过的
	list     *list.List
	mp       map[int]*list.Element
	capacity int
}
type entry struct {
	key, value int
}

//func Constructor(capacity int) LRUCache {
//	return LRUCache{mp: map[int]*list.Element{}, list: list.New(), capacity: capacity}
//}

func (this *LRUCache) Get(key int) int {
	if element, ok := this.mp[key]; ok {
		this.list.MoveToFront(element)
		return element.Value.(entry).value
	} else {
		return -1
	}
}

func (this *LRUCache) Put(key int, value int) {
	if element, ok := this.mp[key]; ok { //已经存在与缓存中 直接修改数值
		this.list.MoveToFront(element)
		element.Value = entry{key: key, value: value}
	} else {
		if this.list.Len() >= this.capacity { //需要移除最久未使用的关键字
			delete(this.mp, this.list.Remove(this.list.Back()).(entry).key)
		}
		this.mp[key] = this.list.PushFront(entry{key: key, value: value})
	}
}

//func reverseList_(head *ListNode) *ListNode {
//	var dfs func(node *ListNode) *ListNode
//	dfs = func(node *ListNode) *ListNode {
//		if node == nil || node.Next == nil {
//			return node
//		}
//		newHead := dfs(node.Next)
//		node.Next.Next = node
//		node.Next = nil
//		return newHead
//	}
//	return dfs(head)
//}

func findKthLargest(nums []int, k int) int {
	hp := &IntHeap{}
	for _, num := range nums {
		heap.Push(hp, num)
		if hp.Len() > k {
			heap.Pop(hp)
		}
	}
	return (*hp)[0]
}
func reverseKGroup2(head *ListNode, k int) *ListNode {
	n := 0
	for p := head; p != nil; p = p.Next {
		n++
	}
	dummy := &ListNode{Next: head}
	temp := dummy
	cur := dummy.Next
	var pre *ListNode
	for i := 0; i < n/k; i++ {
		for j := 0; j < k; j++ {
			nxt := cur.Next
			cur.Next = pre
			pre, cur = cur, nxt
		}
		temp.Next.Next = cur
		newTemp := temp.Next
		temp.Next = pre
		temp = newTemp
	}
	return dummy.Next
}

func maxSubArray_(nums []int) int {
	ans := nums[0]
	for i := 1; i < len(nums); i++ {
		nums[i] = max(nums[i-1]+nums[i], nums[i])
		ans = max(ans, nums[i])
	}
	return ans
}

// 三数之和：双指针+两次去重
func threeSum2(nums []int) (ans [][]int) {
	slices.Sort(nums)
	for i := 0; i < len(nums); i++ {
		n1 := nums[i]
		if (i == 0 || n1 != nums[i-1]) && n1 <= 0 {
			left, right := i+1, len(nums)-1
			for left < right {
				n2, n3 := nums[left], nums[right]
				sum := n1 + n2 + n3
				if sum == 0 {
					ans = append(ans, []int{n1, n2, n3})
					for left < right && nums[left] == n2 {
						left++
					}
					for left < right && nums[right] == n3 {
						right--
					}
				} else if sum > 0 {
					right--
				} else {
					left++
				}
			}
		}
	}
	return
}

//type ListHeap []*ListNode
//
//func (l *ListHeap) Len() int {
//	return len(*l)
//}
//
//func (l *ListHeap) Less(i, j int) bool {
//	return (*l)[i].Val < (*l)[j].Val
//}
//
//func (l *ListHeap) Swap(i, j int) {
//	(*l)[i], (*l)[j] = (*l)[j], (*l)[i]
//}
//
//func (l *ListHeap) Push(x any) {
//	*l = append(*l, x.(*ListNode))
//}
//
//func (l *ListHeap) Pop() any {
//	x := (*l)[(*l).Len()-1]
//	*l = (*l)[:(*l).Len()-1]
//	return x
//}

//func mergeKLists_(lists []*ListNode) *ListNode {
//	dummy := &ListNode{}
//	p := dummy
//	hp := &ListHeap{}
//	for _, l := range lists {
//		if l != nil {
//			heap.Push(hp, l)
//		}
//	}
//	for hp.Len() > 0 {
//		x := heap.Pop(hp).(*ListNode)
//		if x.Next != nil {
//			heap.Push(hp, x.Next)
//		}
//		p.Next = &ListNode{Val: x.Val}
//		p = p.Next
//	}
//	return dummy.Next
//}

// 动态规划法：回文天生具有状态转移的性质
func longestPalindrome(s string) string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(s); i++ {
		dp[i] = make([]bool, len(s))
	}
	ans := ""
	for i := len(dp) - 1; i >= 0; i-- {
		for j := i; j < len(dp); j++ {
			if s[i] == s[j] && (j-i <= 1 || dp[i+1][j-1]) {
				dp[i][j] = true
				if j-i+1 > len(ans) {
					ans = s[i : j+1]
				}
			}
		}
	}
	return ans
}

// 用递归实现层序遍历
func levelOrder__(root *TreeNode) [][]int {
	levels := make([][]int, 0)
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if depth > len(levels) {
			levels = append(levels, make([]int, 0))
		}
		levels[depth-1] = append(levels[depth-1], node.Val)
		dfs(node.Left, depth+1)
		dfs(node.Right, depth+1)
	}
	dfs(root, 1)
	return levels
}

// 搜索旋转排序数组
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		}
		if nums[mid] >= nums[left] {
			if target < nums[left] || target > nums[mid] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		} else {
			if target < nums[mid] || target >= nums[left] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		}
		//如果mid==target return

		//如果mid>=left 说明mid处于第一个序列
		//①如果target<left left=mid+1
		//②如果left<=target<=mid right=mid-1
		//③如果target>mid left=mid+1
		//如果mid<left 说明mid处于第二个序列
		//①如果target<=mid right=mid-1
		//②如果target<=right left=mid+1
		//③如果target>=left right=mid-1
	}
	return -1
}
func merge_(nums1 []int, m int, nums2 []int, n int) {
	i, j := m-1, n-1
	for i >= 0 && j >= 0 {
		if nums1[i] >= nums2[j] {
			nums1[i+j+2] = nums1[i]
			i--
		} else {
			nums1[i+j+2] = nums2[j]
			j--
		}
	}
	if j >= 0 {
		for k := 0; k < j; k++ {
			nums1[k] = nums2[k]
		}
	}
}
func permute__(nums []int) (ans [][]int) {
	used := make([]bool, len(nums))
	var dfs func(i int, path []int)
	dfs = func(i int, path []int) {
		if i == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		for j := 0; j < len(nums); j++ {
			if !used[j] {
				path = append(path, nums[j])
				used[j] = true
				dfs(i+1, path)
				path = path[:len(path)-1]
				used[j] = false
			}
		}
	}
	dfs(0, []int{})
	return
}
func isValid___(s string) bool {
	stack := make([]byte, 0, len(s))
	mp := map[byte]byte{')': '(', ']': '[', '}': '{'}
	for _, ch := range s {
		if ch == '(' || ch == '[' || ch == '{' {
			stack = append(stack, byte(ch))
		} else {
			if len(stack) == 0 || stack[len(stack)-1] != mp[byte(ch)] {
				return false
			} else {
				stack = stack[:len(stack)-1]
			}
		}
	}
	return len(stack) == 0
}

// 动态规划解决：注意只能买卖一次
//func maxProfit_(prices []int) int {
//	//dp[i][0]:不持有股票
//	//dp[i][1]:持有股票
//	dp := make([][2]int, len(prices))
//	dp[0][1] = -prices[0]
//	for i := 1; i < len(prices); i++ {
//		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
//		dp[i][1] = max(-prices[i], dp[i-1][1])
//	}
//	return max(dp[len(prices)-1][0], dp[len(prices)-1][1])
//}

// 对于每一个price 维护其之前的最低价格
func maxProfit_(prices []int) int {
	curMin := prices[0]
	ans := 0
	for i := 1; i < len(prices); i++ {
		curMin = min(curMin, prices[i])
		ans = max(ans, prices[i]-curMin)
	}
	return ans
}
func addStrings(num1 string, num2 string) string {
	byte1, byte2 := []byte(num1), []byte(num2)
	ansByte := make([]byte, max(len(num1), len(num2)))
	carry := 0
	i, j := len(num1)-1, len(num2)-1
	for i >= 0 && j >= 0 {
		val := int(byte1[i]-'0') + int(byte2[j]-'0') + carry
		carry = val / 10
		val %= 10
		ansByte[max(i, j)] = byte(val + '0')
		i--
		j--
	}
	for i >= 0 {
		val := int(byte1[i]-'0') + carry
		carry = val / 10
		val %= 10
		ansByte[i] = byte(val + '0')
		i--
	}
	for j >= 0 {
		val := int(byte2[j]-'0') + carry
		carry = val / 10
		val %= 10
		ansByte[j] = byte(val + '0')
		j--
	}
	if carry != 0 {
		ansByte = append([]byte{'1'}, ansByte...)
	}
	return string(ansByte)
}
func getIntersectionNode_(headA, headB *ListNode) *ListNode {
	mp := make(map[*ListNode]struct{})
	for p := headA; p != nil; p = p.Next {
		mp[p] = struct{}{}
	}
	for p := headB; p != nil; p = p.Next {
		if _, ok := mp[p]; ok {
			return p
		}
	}
	return nil
}
func lowestCommonAncestor_(root, p, q *TreeNode) *TreeNode {
	var dfs func(node *TreeNode) *TreeNode
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil {
			return node
		}
		if node == p || node == q {
			return node
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		if left != nil && right != nil {
			return node
		}
		if left != nil {
			return left
		} else {
			return right
		}
	}
	return dfs(root)
}

//	func reorderList(head *ListNode) {
//		slow, fast := head, head
//		for fast != nil && fast.Next != nil {
//			slow = slow.Next
//			fast = fast.Next.Next
//		}
//		reversed := reverseList(slow.Next)
//		slow.Next = nil
//		for p1, p2 := head, reversed; p2 != nil; {
//			nxt := p2.Next
//			p2.Next = p1.Next
//			p1.Next = p2
//			p1 = p2.Next
//			p2 = nxt
//		}
//	}
func detectCycle_(head *ListNode) *ListNode {
	slow, fast := head, head
	hasCycle := false
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			hasCycle = true
			break
		}
	}
	if !hasCycle {
		return nil
	} else {
		for p := head; p != slow; {
			p = p.Next
			slow = slow.Next
		}
		return slow
	}
}
func discountPrices(sentence string, discount int) string {
	splits := strings.Split(sentence, " ")
	dis := 1 - float64(discount)/100
	for i, split := range splits {
		if split[0] == '$' {
			if val, _ := strconv.Atoi(split[1:]); val > 0 {
				splits[i] = fmt.Sprintf("$%.2f", float64(val)*dis)
			}
		}
	}
	return strings.Join(splits, " ")
}

type ListHeap []*ListNode

func (l *ListHeap) Len() int {
	return len(*l)
}

func (l *ListHeap) Less(i, j int) bool {
	return (*l)[i].Val < (*l)[j].Val
}

func (l *ListHeap) Swap(i, j int) {
	(*l)[i], (*l)[j] = (*l)[j], (*l)[i]
}

func (l *ListHeap) Push(x any) {
	*l = append(*l, x.(*ListNode))
}

func (l *ListHeap) Pop() any {
	x := (*l)[len(*l)-1]
	*l = (*l)[:len(*l)-1]
	return x
}

// 合并K个升序链表 0618
func mergeKLists(lists []*ListNode) *ListNode {
	hp := &ListHeap{}
	for _, l := range lists {
		if l != nil {
			heap.Push(hp, l)
		}
	}
	dummy := &ListNode{}
	p := dummy
	for hp.Len() > 0 {
		x := heap.Pop(hp).(*ListNode)
		if x.Next != nil {
			heap.Push(hp, x.Next)
		}
		p.Next = x
		p = p.Next
	}
	return dummy.Next
}
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	for cur := head; cur != nil; {
		nxt := cur.Next
		cur.Next = pre
		pre, cur = cur, nxt
	}
	return pre
}
func reorderList(head *ListNode) {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	reversed := reverseList(slow.Next)
	slow.Next = nil
	p1, p2 := head, reversed
	for p1 != nil && p2 != nil {
		nxt := p2.Next
		p2.Next = p1.Next
		p1.Next = p2
		p1 = p2.Next
		p2 = nxt
	}
}
func groupAnagrams(strs []string) [][]string {
	mp := make(map[string]int)
	ans := make([][]string, 0, len(strs))
	for _, str := range strs {
		b := []byte(str)
		slices.Sort(b)
		sorted := string(b)
		if i, ok := mp[sorted]; ok {
			ans[i] = append(ans[i], str)
		} else {
			mp[sorted] = len(ans)
			ans = append(ans, []string{str})
		}
	}
	return ans
}
func isPalindrome(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	reversed := reverseList(slow)
	for p1, p2 := head, reversed; p1 != nil && p2 != nil; p1, p2 = p1.Next, p2.Next {
		if p1.Val != p2.Val {
			return false
		}
	}
	return true
}

// 单调栈寻找下一个更大的：单调递减的单调栈
func dailyTemperatures(temperatures []int) []int {
	stack := make([]int, 0, len(temperatures))
	ans := make([]int, len(temperatures))
	for i, t := range temperatures {
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < t {
			ans[stack[len(stack)-1]] = i - stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}

//func invertTree(root *TreeNode) *TreeNode {
//	var dfs func(node *TreeNode)
//	dfs = func(node *TreeNode) {
//		if node == nil {
//			return
//		}
//		node.Left, node.Right = node.Right, node.Left
//		dfs(node.Left)
//		dfs(node.Right)
//	}
//	dfs(root)
//	return root
//}

// 层序遍历实现
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	queue := make([]*TreeNode, 0, 1000)
	queue = append(queue, root)
	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			temp := queue[0]
			temp.Left, temp.Right = temp.Right, temp.Left
			queue = queue[1:]
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
	}
	return root
}
