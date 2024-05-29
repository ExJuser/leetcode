package main

import (
	"container/heap"
	"slices"
)

func maxSatisfied(customers []int, grumpy []int, minutes int) int {
	var ans, cnt int
	for i := 0; i < len(customers); i++ {
		if grumpy[i] == 0 {
			cnt += customers[i]
		}
	}
	for i := 0; i < minutes && i < len(grumpy); i++ {
		if grumpy[i] == 1 {
			cnt += customers[i]
		}
	}
	ans = cnt
	for i := 1; i < len(customers)-minutes+1; i++ {
		if grumpy[i-1] == 1 {
			cnt -= customers[i-1]
		}
		if grumpy[i+minutes-1] == 1 {
			cnt += customers[i+minutes-1]
		}
		ans = max(ans, cnt)
	}
	return ans
}

func maxSum(nums []int, m int, k int) int64 {
	mp := make(map[int]int)
	var sum, ans int64
	for i := 0; i < k; i++ {
		mp[nums[i]]++
		sum += int64(nums[i])
	}
	if len(mp) >= m {
		ans = sum
	}
	for i := 1; i < len(nums)-k+1; i++ {
		mp[nums[i-1]]--
		mp[nums[i+k-1]]++
		sum -= int64(nums[i-1] - nums[i+k-1])
		if cnt, ok := mp[nums[i-1]]; ok && cnt == 0 {
			delete(mp, nums[i-1])
		}
		if len(mp) >= m {
			ans = max(ans, sum)
		}
	}
	return ans
}

// 找出缺失的观测数据
func missingRolls(rolls []int, mean int, n int) []int {
	ans := make([]int, n)
	mRollsSum := 0
	for _, roll := range rolls {
		mRollsSum += roll
	}
	nRollsSum := mean*(n+len(rolls)) - mRollsSum
	if nRollsSum < n || nRollsSum > 6*n {
		return []int{}
	}
	for i := 0; i < len(ans); i++ {
		ans[i] = nRollsSum / n
	}
	for i := 0; i < nRollsSum-(nRollsSum/n)*n; i++ {
		ans[i]++
	}
	return ans
}

// 简单的优先级队列
func minOperations(nums []int, k int) (ans int) {
	hp := &IntHeap{}
	*hp = nums
	heap.Init(hp)
	for ; (*hp)[0] < k; ans++ {
		x, y := heap.Pop(hp).(int), heap.Pop(hp).(int)
		heap.Push(hp, min(x, y)*2+max(x, y))
	}
	return
}

func minStoneSum(piles []int, k int) (ans int) {
	hp := &IntHeap{}
	*hp = piles
	heap.Init(hp)
	for ; k > 0; k-- {
		maxPile := heap.Pop(hp).(int)
		heap.Push(hp, maxPile-maxPile/2)
	}

	for _, pile := range *hp {
		ans += pile
	}
	return
}

type SeatManager struct {
	hp *IntHeap
}

//func Constructor(n int) SeatManager {
//	seats := &IntHeap{}
//	*seats = make([]int, n)
//	for i := 1; i <= n; i++ {
//		(*seats)[i-1] = i
//	}
//	return SeatManager{hp: seats}
//}

func (this *SeatManager) Reserve() int {
	return heap.Pop(this.hp).(int)
}

func (this *SeatManager) Unreserve(seatNumber int) {
	heap.Push(this.hp, seatNumber)
}

type CandidateHeap [][2]int

func (c *CandidateHeap) Len() int {
	return len(*c)
}

func (c *CandidateHeap) Less(i, j int) bool {
	if (*c)[i][0] == (*c)[j][0] {
		return (*c)[i][1] < (*c)[j][1]
	}
	return (*c)[i][0] < (*c)[j][0]
}

func (c *CandidateHeap) Swap(i, j int) {
	(*c)[i], (*c)[j] = (*c)[j], (*c)[i]
}

func (c *CandidateHeap) Push(x any) {
	*c = append(*c, x.([2]int))
}

func (c *CandidateHeap) Pop() any {
	x := (*c)[c.Len()-1]
	*c = (*c)[:c.Len()-1]
	return x
}

func totalCost(costs []int, k int, candidates int) (ans int64) {
	candidateHeap := &CandidateHeap{}
	if candidates*2 < len(costs) {
		for i := 0; i < candidates; i++ {
			heap.Push(candidateHeap, [2]int{costs[i], 0})
			heap.Push(candidateHeap, [2]int{costs[len(costs)-i-1], 1})
		}
	} else {
		for i := 0; i < len(costs); i++ {
			heap.Push(candidateHeap, [2]int{costs[i], 0})
		}
	}
	leftCandidates, left, right := len(costs), candidates, len(costs)-candidates-1
	for ; candidates*2 < leftCandidates && k > 0; k-- {
		heapTop := heap.Pop(candidateHeap).([2]int)
		ans += int64(heapTop[0])
		if heapTop[1] == 0 {
			heap.Push(candidateHeap, [2]int{costs[left], 0})
			left++
		} else {
			heap.Push(candidateHeap, [2]int{costs[right], 1})
			right--
		}
		leftCandidates--
	}
	for ; k > 0; k-- {
		heapTop := heap.Pop(candidateHeap).([2]int)
		ans += int64(heapTop[0])
	}
	return
}

type OrderHeap [][]int

func (o *OrderHeap) Len() int {
	return len(*o)
}

func (o *OrderHeap) Less(i, j int) bool {
	if (*o)[i][1] == (*o)[j][1] {
		return (*o)[i][2] < (*o)[j][2]
	}
	return (*o)[i][1] < (*o)[j][1]
}

func (o *OrderHeap) Swap(i, j int) {
	(*o)[i], (*o)[j] = (*o)[j], (*o)[i]
}

func (o *OrderHeap) Push(x any) {
	*o = append(*o, x.([]int))
}

func (o *OrderHeap) Pop() any {
	x := (*o)[o.Len()-1]
	*o = (*o)[:o.Len()-1]
	return x
}

// 感觉写麻烦了
func getOrder(tasks [][]int) (ans []int) {
	orderHeap := &OrderHeap{}
	tasksWithIndex := make([][]int, len(tasks))
	for i := 0; i < len(tasks); i++ {
		//三个分别是任务id 任务进入队列时间 任务耗时
		tasksWithIndex[i] = []int{i, tasks[i][0], tasks[i][1]}
	}
	slices.SortFunc(tasksWithIndex, func(a, b []int) int {
		if a[1] == b[1] {
			return a[2] - b[2]
		}
		return a[1] - b[1]
	})
	var curTime, finishedTask int
	taskIndex := 0
	for finishedTask < len(tasksWithIndex) {
		for taskIndex < len(tasksWithIndex) && curTime >= tasksWithIndex[taskIndex][1] {
			heap.Push(orderHeap, []int{tasksWithIndex[taskIndex][1], tasksWithIndex[taskIndex][2], tasksWithIndex[taskIndex][0]})
			taskIndex++
		}
		if orderHeap.Len() == 0 {
			curTime = tasksWithIndex[taskIndex][1]
		} else {
			task := heap.Pop(orderHeap).([]int)
			ans = append(ans, task[2])
			curTime += task[1]
			finishedTask++
		}
	}
	return
}

//type IntHeap [][]int
//
//func (h IntHeap) Len() int           { return len(h) }
//func (h IntHeap) Less(i, j int) bool { return h[i][0] < h[j][0] }
//func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
//func (h *IntHeap) Pop() interface{} {
//	old := *h
//	n := len(old)
//	x := old[n-1]
//	*h = old[0 : n-1]
//	return x
//}
//func (h *IntHeap) Push(x interface{}) {
//	*h = append(*h, x.([]int))
//}
//func eatenApples(apples []int, days []int) int {
//	pq := &IntHeap{}
//	i, ans := 0, 0
//	for i < len(apples) || pq.Len() > 0{
//		for pq.Len() > 0 && (*pq)[0][0] <= i {
//			heap.Pop(pq)
//		}
//		if i < len(apples) && apples[i] > 0{
//			heap.Push(pq, []int{i + days[i], apples[i]})
//		}
//		if pq.Len() > 0 {
//			ans++
//			(*pq)[0][1]--
//			if (*pq)[0][1] == 0{
//				heap.Pop(pq)
//			}
//		}
//		i++
//	}
//	return ans
//}

func maximumSubarraySum(nums []int, k int) (ans int64) {
	mp := make(map[int]int)
	var sum int
	for i := 0; i < k; i++ {
		mp[nums[i]]++
		sum += nums[i]
	}
	if len(mp) == k {
		ans = int64(sum)
	}
	for i := 1; i < len(nums)-k+1; i++ {
		mp[nums[i-1]]--
		mp[nums[i+k-1]]++
		sum -= nums[i-1] - nums[i+k-1]
		if mp[nums[i-1]] == 0 {
			delete(mp, nums[i-1])
		}
		if len(mp) == k {
			ans = max(ans, int64(sum))
		}
	}
	return ans
}

func maxScore(cardPoints []int, k int) int {
	n := len(cardPoints) - k
	cardSum := 0
	for _, card := range cardPoints {
		cardSum += card
	}
	var sum int
	for i := 0; i < n; i++ {
		sum += cardPoints[i]
	}
	minSum := sum
	for i := 1; i < len(cardPoints)-n+1; i++ {
		sum -= cardPoints[i-1] - cardPoints[i+n-1]
		minSum = min(minSum, sum)
	}
	return cardSum - minSum
}

func searchMatrix(matrix [][]int, target int) bool {
	for _, row := range matrix {
		if target < row[0] {
			return false
		} else if target >= row[0] && target <= row[len(row)-1] {
			if _, b := slices.BinarySearch(row, target); b {
				return true
			}
		}
		continue
	}
	return false
}

func rotate2(matrix [][]int) {
	for i := 0; i < len(matrix); i++ {
		for j := i + 1; j < len(matrix[i]); j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
	for _, row := range matrix {
		slices.Reverse(row)
	}
}

// O(m+n)的空间复杂度
func setZeroes(matrix [][]int) {
	zeroRow := make(map[int]struct{})
	zeroCol := make(map[int]struct{})
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if matrix[i][j] == 0 {
				zeroRow[i] = struct{}{}
				zeroCol[j] = struct{}{}
			}
		}
	}
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			_, ok1 := zeroRow[i]
			_, ok2 := zeroCol[j]
			if ok1 || ok2 {
				matrix[i][j] = 0
			}
		}
	}
}

// 合并区间
func merge1(intervals [][]int) (ans [][]int) {
	slices.SortFunc(intervals, func(a, b []int) int {
		if a[0] == b[0] {
			return a[1] - b[1]
		}
		return a[0] - b[0]
	})
	pre := intervals[0]
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] > pre[1] {
			ans = append(ans, pre)
			pre = cur
		} else {
			pre[1] = max(pre[1], cur[1])
		}
	}
	ans = append(ans, pre)
	return
}

func rotate1(nums []int, k int) {
	slices.Reverse(nums)
	slices.Reverse(nums[:k])
	slices.Reverse(nums[k:])
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
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

func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	for cur := head; cur != nil; {
		nxt := cur.Next
		cur.Next = pre
		pre, cur = cur, nxt
	}
	return pre
}
func isPalindrome1(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	for p, q := head, reverseList(slow); p != nil && q != nil; p, q = p.Next, q.Next {
		if p.Val != q.Val {
			return false
		}
	}
	return true
}
func hasCycle(head *ListNode) bool {
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

func detectCycle(head *ListNode) *ListNode {
	if !hasCycle(head) {
		return nil
	}
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			break
		}
	}
	for p := head; p != slow; {
		p = p.Next
		slow = slow.Next
	}
	return slow
}
func findPeaks(mountain []int) (ans []int) {
	for i := 1; i < len(mountain)-1; i++ {
		if mountain[i] > mountain[i-1] && mountain[i] > mountain[i+1] {
			ans = append(ans, i)
		}
	}
	return
}
