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

//	func reverseList(head *ListNode) *ListNode {
//		var pre *ListNode
//		for cur := head; cur != nil; {
//			nxt := cur.Next
//			cur.Next = pre
//			pre, cur = cur, nxt
//		}
//		return pre
//	}
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

// 写的太丑陋了
//func minSwaps(nums []int) int {
//	var zero, ans, zeroCnt, oneCnt int
//	for _, num := range nums {
//		if num == 0 {
//			zero++
//		}
//	}
//	for i := 0; i < zero; i++ {
//		if nums[i] == 1 {
//			oneCnt++
//		}
//	}
//	ans = oneCnt
//	for i := 1; i < len(nums)-zero+1; i++ {
//		if nums[i-1] == 1 {
//			oneCnt--
//		}
//		if nums[i+zero-1] == 1 {
//			oneCnt++
//		}
//		ans = min(ans, oneCnt)
//	}
//	for i := 0; i < len(nums)-zero; i++ {
//		if nums[i] == 0 {
//			zeroCnt++
//		}
//	}
//	ans = min(ans, zeroCnt)
//	for i := 1; i < zero+1; i++ {
//		if nums[i-1] == 0 {
//			zeroCnt--
//		}
//		if nums[i+len(nums)-zero-1] == 0 {
//			zeroCnt++
//		}
//		ans = min(ans, zeroCnt)
//	}
//	return ans
//}

func minSwaps(nums []int) int {
	var oneCnt, zeros, ans int
	for _, num := range nums {
		if num == 1 {
			oneCnt++
		}
	}
	circular := append(nums, nums...)
	for i := 0; i < oneCnt; i++ {
		if nums[i] == 0 {
			zeros++
		}
	}
	ans = zeros
	for i := 1; i < len(nums); i++ {
		if circular[i-1] == 0 {
			zeros--
		}
		if circular[i+oneCnt-1] == 0 {
			zeros++
		}
		ans = min(ans, zeros)
	}
	return ans
}
func checkInclusion(s1 string, s2 string) bool {
	if len(s1) > len(s2) {
		return false
	}
	mp := make(map[byte]int)
	for _, char := range s1 {
		mp[byte(char)]++
	}
	for i := 0; i < len(s1); i++ {
		mp[s2[i]]--
		if mp[s2[i]] == 0 {
			delete(mp, s2[i])
		}
	}
	if len(mp) == 0 {
		return true
	}
	for i := 1; i < len(s2)-len(s1)+1; i++ {
		mp[s2[i-1]]++
		if mp[s2[i-1]] == 0 {
			delete(mp, s2[i-1])
		}
		mp[s2[i+len(s1)-1]]--
		if mp[s2[i+len(s1)-1]] == 0 {
			delete(mp, s2[i+len(s1)-1])
		}
		if len(mp) == 0 {
			return true
		}
	}
	return false
}

func findAnagrams(s string, p string) (ans []int) {
	if len(s) < len(p) {
		return
	}
	mp := make(map[byte]int)
	for _, char := range p {
		mp[byte(char)]++
	}
	for i := 0; i < len(p); i++ {
		mp[s[i]]--
		if mp[s[i]] == 0 {
			delete(mp, s[i])
		}
	}
	if len(mp) == 0 {
		ans = append(ans, 0)
	}
	for i := 1; i < len(s)-len(p)+1; i++ {
		mp[s[i-1]]++
		if mp[s[i-1]] == 0 {
			delete(mp, s[i-1])
		}
		mp[s[i+len(p)-1]]--
		if mp[s[i+len(p)-1]] == 0 {
			delete(mp, s[i+len(p)-1])
		}
		if len(mp) == 0 {
			ans = append(ans, i)
		}
	}
	return
}

type FloatHeap []float64

func (f *FloatHeap) Len() int {
	return len(*f)
}

func (f *FloatHeap) Less(i, j int) bool {
	return (*f)[i] > (*f)[j]
}

func (f *FloatHeap) Swap(i, j int) {
	(*f)[i], (*f)[j] = (*f)[j], (*f)[i]
}

func (f *FloatHeap) Push(x any) {
	*f = append(*f, x.(float64))
}

func (f *FloatHeap) Pop() any {
	x := (*f)[f.Len()-1]
	*f = (*f)[:f.Len()-1]
	return x
}

func halveArray(nums []int) int {
	var arrSum float64
	var ans int
	for _, num := range nums {
		arrSum += float64(num)
	}
	halfSum := arrSum / 2
	hp := &FloatHeap{}
	for _, num := range nums {
		heap.Push(hp, float64(num))
	}
	for arrSum > halfSum {
		half := heap.Pop(hp).(float64) / 2
		arrSum -= half
		heap.Push(hp, half)
		ans++
	}
	return ans
}
func maximumProduct(nums []int, k int) int {
	hp := &IntHeap{}
	*hp = nums
	heap.Init(hp)
	for ; k > 0; k-- {
		heap.Push(hp, heap.Pop(hp).(int)+1)
	}
	ans := 1
	for _, num := range *hp {
		ans = (ans * num) % (1e9 + 7)
	}
	return ans
}

// 不定长滑动窗口
//func lengthOfLongestSubstring(s string) (ans int) {
//	left := 0
//	mp := make(map[byte]int)
//	for right := 0; right < len(s); right++ {
//		mp[s[right]]++
//		for len(mp) < right-left+1 {
//			mp[s[left]]--
//			if mp[s[left]] == 0 {
//				delete(mp, s[left])
//			}
//			left++
//		}
//		ans = max(ans, right-left+1)
//	}
//	return
//}

// 完全背包问题：先遍历背包和物品都可以
func combinationSum41(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 0; i <= target; i++ {
		for j := 0; j < len(nums); j++ {
			if i >= nums[j] {
				dp[i] += dp[i-nums[j]]
			}
		}
	}
	return dp[target]
}
func countGoodStrings(low int, high int, zero int, one int) int {
	dp := make([]int, high+1)
	dp[zero] += 1
	dp[one] += 1
	for i := min(zero, one); i <= high; i++ {
		if i+zero <= high {
			dp[i+zero] = (dp[i+zero] + dp[i]) % (1e9 + 7)
		}
		if i+one <= high {
			dp[i+one] = (dp[i+one] + dp[i]) % (1e9 + 7)
		}
	}
	ans := 0
	for i := low; i <= high; i++ {
		ans = (ans + dp[i]) % (1e9 + 7)
	}
	return ans
}

// 相邻的数字一定无法同时删除 而且相同的数字要么同时被删除要么同时取得
// 转化为打家劫舍问题
func deleteAndEarn(nums []int) int {
	rob := make([]int, slices.Max(nums)+1)
	for _, num := range nums {
		rob[num] += num
	}
	dp := make([]int, len(rob))
	if len(rob) == 1 {
		return rob[0]
	}
	dp[0] = rob[0]
	dp[1] = max(rob[0], rob[1])
	for i := 2; i < len(rob); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+rob[i])
	}
	return dp[len(rob)-1]
}
func maxSubArray3(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	ans := nums[0]
	for i := 1; i < len(dp); i++ {
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		ans = max(ans, dp[i])
	}
	return ans
}
func maximumCostSubstring(s string, chars string, vals []int) int {
	dp := make([]int, len(s))
	valMap := make(map[byte]int)
	for i, char := range chars {
		valMap[byte(char)] = vals[i]
	}
	if _, ok := valMap[s[0]]; ok {
		dp[0] = max(0, valMap[s[0]])
	} else {
		dp[0] = int(s[0] - 'a' + 1)
	}
	ans := dp[0]
	for i := 1; i < len(s); i++ {
		if _, ok := valMap[s[i]]; ok {
			dp[i] = max(valMap[s[i]], dp[i-1]+valMap[s[i]], 0)
		} else {
			dp[i] = max(dp[i-1]+int(s[i]-'a'+1), int(s[i]-'a'+1), 0)
		}
		ans = max(ans, dp[i])
	}
	return ans
}
func maxAbsoluteSum(nums []int) int {
	maxDp := make([]int, len(nums))
	minDp := make([]int, len(nums))
	maxDp[0] = nums[0]
	minDp[0] = nums[0]
	ans := max(Abs(maxDp[0]), Abs(minDp[0]), 0)
	for i := 1; i < len(nums); i++ {
		maxDp[i] = max(maxDp[i-1]+nums[i], nums[i])
		minDp[i] = min(minDp[i-1]+nums[i], nums[i])
		ans = max(ans, Abs(maxDp[i]), Abs(minDp[i]))
	}
	return ans
}
func kConcatenationMaxSum(arr []int, k int) int {
	var maxSubArray func(nums []int) int
	maxSubArray = func(nums []int) int {
		ans := nums[0]
		for i := 1; i < len(nums); i++ {
			nums[i] = max(nums[i], (nums[i-1]+nums[i])%(1e9+7))
			ans = max(ans, nums[i])
		}
		return max(ans, 0)
	}
	newArr := make([]int, 0, 3*len(arr))
	for ; k > 0; k-- {
		newArr = append(newArr, arr...)
	}
	return maxSubArray(newArr)
}

// 层序遍历
func levelOrder1(root *TreeNode) (ans [][]int) {
	if root == nil {
		return
	}
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		size := len(queue)
		level := make([]int, 0, size)
		for i := 0; i < size; i++ {
			temp := queue[0]
			queue = queue[1:]
			level = append(level, temp.Val)
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
		ans = append(ans, level)
	}
	return
}
func lowestCommonAncestor1(root, p, q *TreeNode) *TreeNode {
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
		} else if left != nil {
			return left
		} else {
			return right
		}
	}
	return dfs(root)
}
func zigzagLevelOrder(root *TreeNode) (ans [][]int) {
	if root == nil {
		return
	}
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)
	depth := 0
	for ; len(queue) > 0; depth++ {
		size := len(queue)
		list := make([]int, 0, size)
		for i := 0; i < size; i++ {
			temp := queue[0]
			list = append(list, temp.Val)
			queue = queue[1:]
			if temp.Left != nil {
				queue = append(queue, temp.Left)
			}
			if temp.Right != nil {
				queue = append(queue, temp.Right)
			}
		}
		if depth%2 != 0 {
			slices.Reverse(list)
		}
		ans = append(ans, list)
	}
	return
}

func inorderTraversal(root *TreeNode) (ans []int) {
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		ans = append(ans, node.Val)
		dfs(node.Right)
	}
	dfs(root)
	return
}

func rightSideView(root *TreeNode) (ans []int) {
	var dfs func(node *TreeNode, depth int)
	dfs = func(node *TreeNode, depth int) {
		if node == nil {
			return
		}
		if depth == len(ans) {
			ans = append(ans, node.Val)
		}
		//可能最右节点在左子树
		dfs(node.Right, depth+1)
		dfs(node.Left, depth+1)
	}
	dfs(root, 0)
	return
}

//func rightSideView(root *TreeNode) (ans []int) {
//	levels := levelOrder(root)
//	for _, level := range levels {
//		ans = append(ans, level[len(level)-1])
//	}
//	return
//}

func maxDepth2(root *TreeNode) int {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		return max(dfs(node.Left), dfs(node.Right)) + 1
	}
	return dfs(root)
}

//func isSameTree(p *TreeNode, q *TreeNode) bool {
//	var dfs func(p, q *TreeNode) bool
//	dfs = func(p, q *TreeNode) bool {
//		if p == nil || q == nil {
//			if p == nil && q == nil {
//				return true
//			}
//			return false
//		}
//		return p.Val == q.Val && dfs(p.Left, q.Right) && dfs(p.Right, q.Left)
//	}
//	return dfs(p, q)
//}

func isSymmetric1(root *TreeNode) bool {
	return isSameTree(root.Left, root.Right)
}
func sumNumbers(root *TreeNode) (ans int) {
	var dfs func(node *TreeNode, sum int)
	dfs = func(node *TreeNode, sum int) {
		if node == nil {
			return
		}
		sum = sum*10 + node.Val
		//叶子结点
		if node.Left == nil && node.Right == nil {
			ans += sum
		}
		dfs(node.Left, sum)
		dfs(node.Right, sum)
	}
	dfs(root, 0)
	return
}

// 所有节点的左右子树深度差不超过1
func isBalanced1(root *TreeNode) bool {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		if left == -1 {
			return -1
		}
		right := dfs(node.Right)
		if right == -1 || Abs(left-right) > 1 {
			return -1
		}
		return max(left, right) + 1
	}
	return dfs(root) != -1
}
func diameterOfBinaryTree1(root *TreeNode) (ans int) {
	var dfs func(node *TreeNode) int
	dfs = func(node *TreeNode) int {
		//对一个节点来说以它为拐点的直径为左子树长度+右子树长度
		//而它需要返回给父结点的是左右子树中最长的那一个
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		ans = max(ans, left+right)
		return max(left, right) + 1
	}
	dfs(root)
	return
}
