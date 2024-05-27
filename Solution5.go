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
