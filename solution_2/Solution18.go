package main

import (
	"container/heap"
	"math"
	"math/bits"
	"slices"
	"sort"
)

// 动态规划 超出内存限制
func minDays(n int) int {
	//dp数组含义 吃掉i个橘子所需要的最小天数
	dp := make([]int, n+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = math.MaxInt
	}
	dp[1] = 1
	for i := 1; i <= n; i++ {
		if i+1 <= n {
			dp[i+1] = min(dp[i+1], dp[i]+1)
		}
		if i*2 <= n {
			dp[i*2] = min(dp[i*2], dp[i]+1)
		}
		if i*3 <= n {
			dp[i*3] = min(dp[i*3], dp[i]+1)
		}
	}
	return dp[n]
}

// 回溯 超时
func minDays2(n int) int {
	res := make([]int, 0)
	//i是天数，n是剩余的橘子数量
	var dfs func(i, n int)
	dfs = func(i, n int) {
		if n <= 0 {
			res = append(res, i)
			return
		}
		dfs(i+1, n-1)
		if n%2 == 0 {
			dfs(i+1, n-n/2)
		}
		if n%3 == 0 {
			dfs(i+1, n-2*(n/3))
		}
	}
	dfs(0, n)
	slices.Sort(res)
	return res[0]
}

// 动态规划的数据量过大时 直接申请巨大的dp数组可能会超内存 可以使用map
func minDays3(n int) int {
	dp := make(map[int]int)
	dp[0] = 0
	dp[1] = 1

	var dfs func(n int) int
	dfs = func(n int) int {
		if days, ok := dp[n]; ok {
			return days
		}
		//n%2和n%3即一个一个吃
		//例如有17个橘子
		//17=>16 1
		//17=>15 2
		dp[n] = min(dfs(n/2)+1+n%2, dfs(n/3)+1+n%3)
		return dp[n]
	}
	return dfs(n)
}
func sortArrayByParityII(nums []int) []int {
	even, odd := 0, 1
	for even < len(nums) {
		if nums[even]%2 != 0 {
			for odd < len(nums) && nums[odd]%2 != 0 {
				odd += 2
			}
			nums[odd], nums[even] = nums[even], nums[odd]
		}
		even += 2
	}
	return nums
}
func sumIndicesWithKSetBits(nums []int, k int) int {
	ans := 0
	for i, num := range nums {
		if bits.OnesCount(uint(i)) == k {
			ans += num
		}
	}
	return ans
}

// 横着计算 单调栈
func trap2(height []int) int {
	ans := 0
	stack := make([]int, 0, len(height))
	for i, h := range height {
		for len(stack) > 0 && height[stack[len(stack)-1]] <= h {
			baseHeight := height[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			if len(stack) > 1 {
				ans += (min(h, height[stack[len(stack)-1]]) - baseHeight) * (i - stack[len(stack)-1] - 1)
			}
		}
		stack = append(stack, i)
	}
	return ans
}

// 竖着计算 前后缀分解
func trap3(height []int) int {
	pre := make([]int, len(height))
	for i := 1; i < len(height); i++ {
		pre[i] = max(pre[i-1], height[i-1])
	}
	suf := make([]int, len(height))
	for i := len(height) - 2; i >= 0; i-- {
		suf[i] = max(suf[i+1], height[i+1])
	}
	ans := 0
	for i, h := range height {
		rain := min(pre[i], suf[i]) - h
		if rain > 0 {
			ans += rain
		}
	}
	return ans
}

// 双指针是时间空间上的最优解
// 一个位置的接水量：左侧最大值和右侧最大值之间的最小值决定
func trap4(height []int) (ans int) {
	left, right, preMax, sufMax := 0, len(height)-1, 0, 0
	for left < right {
		preMax = max(preMax, height[left])
		sufMax = max(sufMax, height[right])
		//哪一侧更小就先结算哪一边
		if preMax < sufMax {
			ans += preMax - height[left]
			left++
		} else {
			ans += sufMax - height[right]
			right--
		}
	}
	return
}
func numRescueBoats(people []int, limit int) (cnt int) {
	slices.Sort(people)
	for left, right := 0, len(people)-1; left <= right; right, cnt = right-1, cnt+1 {
		if people[left]+people[right] <= limit {
			left++
		}
	}
	return
}
func maxArea2(height []int) int {
	left := 0
	right := len(height) - 1
	m := 0
	for left < right {
		area := (right - left) * min(height[left], height[right])
		m = max(m, area)
		//对于短的这一根线(假设为左) 若其右边的线比它短 容器宽度变小 高度也变小 容量一定变小
		//若其右边的线比它长 则宽度变小 高度不变 容量同样变小
		//因此对于短的这一根线 无论如何他都无法构成更大的容器 直接++
		if height[left] <= height[right] {
			left++
		} else {
			right--
		}
	}
	return m
}

// 二分答案法
func findRadius(houses []int, heaters []int) int {
	var check func(radius int) bool
	check = func(radius int) bool {
		i, j := 0, 0
		for i < len(houses) && j < len(heaters) {
			//当前的房子可以被取暖器覆盖到
			//后面的房子一定离当前的取暖器及当前取暖器之后的取暖器更近
			//双指针不回退 如果回退一定是思路错了
			if heaters[j]-radius <= houses[i] && houses[i] <= heaters[j]+radius {
				i++
			} else {
				j++
			}
		}
		if i < len(houses) {
			return false
		}
		return true
	}
	slices.Sort(houses)
	slices.Sort(heaters)
	maxRadius := int(1e9)
	return sort.Search(maxRadius, check)
}

// 和第一个重复的数类似 交换位置
func firstMissingPositive(nums []int) int {
	for i := 0; i < len(nums); i++ {
		for nums[i] > 0 && nums[i] <= len(nums) && nums[i] != i+1 && nums[nums[i]-1] != nums[i] {
			nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
		}
	}
	for i, num := range nums {
		if num != i+1 {
			return i + 1
		}
	}
	return len(nums) + 1
}
func findDuplicate2(nums []int) int {
	for i := 0; i < len(nums); i++ {
		for nums[i] != i+1 {
			if nums[nums[i]-1] == nums[i] {
				return nums[i]
			}
			nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
		}
	}
	return -1
}
func findDuplicate3(nums []int) int {
	hash := make(map[int]struct{})
	for _, num := range nums {
		if _, ok := hash[num]; ok {
			return num
		} else {
			hash[num] = struct{}{}
		}
	}
	return -1
}

// 滑动窗口
func lengthOfLongestSubstring(s string) int {
	count := make(map[byte]int)
	ans := 0
	for left, right := 0, 0; right < len(s); right++ {
		count[s[right]]++
		for right-left+1 > len(count) {
			count[s[left]]--
			if count[s[left]] == 0 {
				delete(count, s[left])
			}
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}
func lengthOfLongestSubstring2(s string) int {
	ans := 0
	hash := make(map[byte]int)
	for left, right := 0, 0; right < len(s); right++ {
		hash[s[right]]++
		for right-left+1 > len(hash) {
			hash[s[left]]--
			if hash[s[left]] == 0 {
				delete(hash, s[left])
			}
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}
func longestSubstring(s string, k int) int {
	var longestSubstringN func(n int) int
	longestSubstringN = func(n int) int {
		cnt := make(map[byte]int)
		satisfy := 0
		ans := 0
		for left, right := 0, 0; right < len(s); right++ {
			cnt[s[right]]++
			if cnt[s[right]] == k {
				satisfy++
			}
			for len(cnt) > n {
				cnt[s[left]]--
				if cnt[s[left]] == k-1 {
					satisfy--
				}
				if cnt[s[left]] == 0 {
					delete(cnt, s[left])
				}
				left++
			}
			if satisfy == n {
				ans = max(ans, right-left+1)
			}
		}
		return ans
	}
	ans := 0
	for i := 1; i <= 26; i++ {
		ans = max(ans, longestSubstringN(i))
	}
	return ans
}

// 假设只有一个字符、两个字符...
func longestSubstring2(s string, k int) int {
	ans := 0
	for i := 1; i <= 26; i++ {
		hash := make(map[byte]int)
		satisfy := 0
		for left, right := 0, 0; right < len(s); right++ {
			hash[s[right]]++
			if hash[s[right]] == k {
				satisfy++
			}
			for len(hash) > i {
				hash[s[left]]--
				if hash[s[left]] == k-1 {
					satisfy--
				}
				if hash[s[left]] == 0 {
					delete(hash, s[left])
				}
				left++
			}
			if satisfy == i && len(hash) == i {
				ans = max(ans, right-left+1)
			}
		}
	}
	return ans
}

// 数组代替哈希表
func longestSubstring3(s string, k int) int {
	ans := 0
	for i := 1; i <= 26; i++ {
		//满足了i中的几个
		satisfy := 0
		//当前遇到了多少个字符
		cnt := 0
		appearTimes := make([]int, 26)
		for left, right := 0, 0; right < len(s); right++ {
			appearTimes[s[right]-'a']++
			if appearTimes[s[right]-'a'] == 1 {
				cnt++
			}
			if appearTimes[s[right]-'a'] == k {
				satisfy++
			}
			for cnt > i {
				appearTimes[s[left]-'a']--
				if appearTimes[s[left]-'a'] == 0 {
					cnt--
				}
				if appearTimes[s[left]-'a'] == k-1 {
					satisfy--
				}
				left++
			}
			if satisfy == i && cnt == i {
				ans = max(ans, right-left+1)
			}
		}
	}
	return ans
}

// 不用哈希表
func lengthOfLongestSubstring3(s string) int {
	//无重复字符意味着 字符的种类等于字符串长度
	appearTimes := make([]int, 128)
	cnt := 0
	ans := 0
	for left, right := 0, 0; right < len(s); right++ {
		appearTimes[s[right]]++
		if appearTimes[s[right]] == 1 {
			cnt++
		}
		for cnt < right-left+1 {
			appearTimes[s[left]]--
			if appearTimes[s[left]] == 0 {
				cnt--
			}
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}
func maxSubArray(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	ans := dp[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(nums[i], nums[i]+dp[i-1])
		ans = max(ans, dp[i])
	}
	return ans
}

// 包含负数 不具备单调性 因此无法使用滑动窗口
func maxSubArray2(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	ans := dp[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(nums[i], dp[i-1]+nums[i])
		ans = max(ans, dp[i])
	}
	return ans
}
func prefixSum(nums []int) []int {
	ans := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		ans[i+1] = nums[i] + ans[i]
	}
	return ans
}

// 返回具有最大和的子数组 不仅仅是最大和
func maxSubArray3(nums []int) (int, []int) {
	ans := nums[0]
	sub := []int{nums[0]}
	pre := prefixSum(nums)
	sum := 0
	for left, right := 0, 0; right < len(nums); right++ {
		sum += nums[right]
		if sum > ans {
			ans = sum
			sub = nums[left : right+1]
		}
		if pre[right+1]-pre[left] <= 0 {
			left = right + 1
			sum = 0
		}
	}
	return ans, sub
}

// 用来当做对数器验证方法的暴力解
func maxSubArray4(nums []int) (int, []int) {
	var getArraySum func(left, right int) int
	getArraySum = func(left, right int) int {
		ans := 0
		for i := left; i <= right; i++ {
			ans += nums[i]
		}
		return ans
	}
	ans := nums[0]
	sub := []int{nums[0]}
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums); j++ {
			sum := getArraySum(i, j)
			if sum > ans {
				ans = sum
				sub = nums[i : j+1]
			}
		}
	}
	return ans, sub
}

// 迭代法:反转链表
//func reverseList(head *ListNode) *ListNode {
//	cur := head
//	var pre *ListNode
//	for cur != nil {
//		next := cur.Next
//		cur.Next = pre
//		pre, cur = cur, next
//	}
//	return pre
//}
//
//// 递归法
//func reverseList2(head *ListNode) *ListNode {
//	if head == nil || head.Next == nil {
//		return head
//	}
//	newHead := reverseList2(head.Next)
//	head.Next.Next = head
//	head.Next = nil
//	return newHead
//}
//
//// 反转链表2：要点是需要找到翻转前的那一个节点
//func reverseBetween(head *ListNode, left int, right int) *ListNode {
//	dummy := &ListNode{Next: head}
//	p := dummy
//	for i := 0; i < left-1; i++ {
//		p = p.Next
//	}
//	var pre *ListNode
//	temp := p
//	cur := p.Next
//	for i := left; i <= right; i++ {
//		next := cur.Next
//		cur.Next = pre
//		pre, cur = cur, next
//	}
//	temp.Next.Next = cur
//	temp.Next = pre
//	return dummy.Next
//}
//
//// k个一组反转链表
//func reverseKGroup(head *ListNode, k int) *ListNode {
//	//先遍历一遍得到节点总数
//	cnt := 0
//	for p := head; p != nil; p = p.Next {
//		cnt++
//	}
//	dummy := &ListNode{Next: head}
//	cur := dummy.Next
//	temp := dummy
//	var pre *ListNode
//	for i := 0; i < cnt/k; i++ {
//		//写在里面效果一样
//		//var pre *ListNode
//		for j := 0; j < k; j++ {
//			next := cur.Next
//			cur.Next = pre
//			pre, cur = cur, next
//		}
//		temp.Next.Next = cur
//		temp.Next = pre
//		for j := 0; j < k; j++ {
//			temp = temp.Next
//		}
//	}
//	return dummy.Next
//}

type DoubleListNode struct {
	Key  int
	Val  int
	next *DoubleListNode
	prev *DoubleListNode
}
type LRUCache struct {
	hash      map[int]*DoubleListNode
	dummyHead *DoubleListNode
	dummyTail *DoubleListNode
	capacity  int
}

func Constructor(capacity int) LRUCache {
	dummyHead := &DoubleListNode{}
	dummyTail := &DoubleListNode{}
	cache := LRUCache{
		hash:      make(map[int]*DoubleListNode),
		dummyHead: dummyHead,
		dummyTail: dummyTail,
		capacity:  capacity,
	}
	cache.dummyHead.next = cache.dummyTail
	cache.dummyTail.prev = cache.dummyHead
	return cache
}

func (this *LRUCache) Get(key int) int {
	if node, ok := this.hash[key]; ok {
		this.moveToEnd(node)
		return node.Val
	} else {
		return -1
	}
}

func (this *LRUCache) Put(key int, value int) {
	if node, ok := this.hash[key]; ok {
		node.Val = value
		this.moveToEnd(node)
	} else {
		newNode := &DoubleListNode{
			Key:  key,
			Val:  value,
			prev: this.dummyTail.prev,
			next: this.dummyTail,
		}
		this.hash[key] = newNode
		this.dummyTail.prev.next = newNode
		this.dummyTail.prev = newNode
		if len(this.hash) > this.capacity {
			delete(this.hash, this.dummyHead.next.Key)
			this.dummyHead.next = this.dummyHead.next.next
			this.dummyHead.next.prev = this.dummyHead.next.prev.prev
		}
	}
}

func (this *LRUCache) moveToEnd(node *DoubleListNode) {
	node.prev.next = node.next
	node.next.prev = node.prev
	node.prev = this.dummyTail.prev
	node.next = this.dummyTail
	this.dummyTail.prev.next = node
	this.dummyTail.prev = node
}

type IntHeap []int

func (h *IntHeap) Len() int {
	return len(*h)
}

func (h *IntHeap) Less(i, j int) bool {
	return (*h)[i] < (*h)[j]
}

func (h *IntHeap) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

func (h *IntHeap) Push(x any) {
	*h = append(*h, x.(int))
}

func (h *IntHeap) Pop() any {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}

// 小根堆实现
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

// 合并区间
func merge(intervals [][]int) (ans [][]int) {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	prev := intervals[0]
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if prev[1] < cur[0] {
			ans = append(ans, prev)
			prev = cur
		} else {
			prev[1] = max(prev[1], cur[1])
		}
	}
	ans = append(ans, prev)
	return
}

// 左边界排序
func eraseOverlapIntervals(intervals [][]int) int {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[0] - b[0]
	})
	prev := intervals[0]
	cnt := 0
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] >= prev[1] {
			prev = cur
			continue
		} else {
			cnt++
			if prev[1] > cur[1] {
				prev = cur
			}
		}
	}
	return cnt
}

// 对右边界排序代码更简单
func eraseOverlapIntervals2(intervals [][]int) int {
	slices.SortFunc(intervals, func(a, b []int) int {
		return a[1] - b[1]
	})
	prev := intervals[0]
	cnt := 0
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if cur[0] < prev[1] {
			cnt++
		} else {
			prev = cur
		}
	}
	return cnt
}

// 左边界排序
func findMinArrowShots(points [][]int) int {
	slices.SortFunc(points, func(a, b []int) int {
		return a[0] - b[0]
	})
	prev := points[0]
	cnt := 0
	for i := 1; i < len(points); i++ {
		cur := points[i]
		if cur[0] > prev[1] {
			cnt++
			prev = cur
		} else {
			if cur[1] < prev[1] {
				prev = cur
			}
		}
	}
	return cnt + 1
}

// 对右边界排序代码更简单
func findMinArrowShots2(points [][]int) int {
	slices.SortFunc(points, func(a, b []int) int {
		return a[1] - b[1]
	})
	prev := points[0]
	cnt := 0
	for i := 1; i < len(points); i++ {
		cur := points[i]
		if cur[0] > prev[1] {
			cnt++
			prev = cur
		}
	}
	return cnt + 1
}

func maxNumberOfAlloys(n int, k int, budget int, composition [][]int, stock []int, cost []int) int {
	var check func(num int) bool
	check = func(num int) bool {
		for i := 0; i < k; i++ {
			price := 0
			for j := 0; j < n; j++ {
				price += max(0, num*composition[i][j]-stock[j]) * cost[j]
				if price > budget {
					continue
				}
			}
			if price <= budget {
				return true
			}
		}
		return false
	}
	left, right := 0, budget+slices.Max(stock)
	ans := 0
	for left <= right {
		mid := left + (right-left)/2
		if check(mid) {
			ans = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return ans
}

// 二维dp 同时维护最大值最小值 防止负负得正
func maxProduct(nums []int) int {
	dp := make([][2]int, len(nums))
	dp[0][0] = nums[0]
	dp[0][1] = nums[0]
	ans := nums[0]
	for i := 1; i < len(nums); i++ {
		dp[i][0] = max(dp[i-1][0]*nums[i], dp[i-1][1]*nums[i], nums[i])
		dp[i][1] = min(dp[i-1][0]*nums[i], dp[i-1][1]*nums[i], nums[i])
		ans = max(ans, dp[i][0])
	}
	return ans
}
func main() {
	findMinArrowShots2(nil)
}
