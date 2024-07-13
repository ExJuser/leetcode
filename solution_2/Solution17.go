package main

import (
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
)

//type IntHeap []int
//
//func (h IntHeap) Len() int {
//	return len(h)
//}
//
//func (h IntHeap) Less(i, j int) bool {
//	return h[i] > h[j]
//}
//
//func (h IntHeap) Swap(i, j int) {
//	h[i], h[j] = h[j], h[i]
//}
//
//func (h *IntHeap) Push(x any) {
//	*h = append(*h, x.(int))
//}
//
//func (h *IntHeap) Pop() any {
//	x := (*h)[h.Len()-1]
//	*h = (*h)[:h.Len()-1]
//	return x
//}

// 优先队列超时
//
//	func maxRunTime(n int, batteries []int) int64 {
//		var cnt int64 = 0
//		hp := &IntHeap{}
//		*hp = batteries
//		heap.Init(hp)
//		for {
//			curBatteries := make([]int, 0, n)
//			for i := 0; i < n; i++ {
//				battery := heap.Pop(hp).(int)
//				if battery <= 0 {
//					return cnt
//				} else {
//					curBatteries = append(curBatteries, battery)
//				}
//			}
//			cnt++
//			for _, battery := range curBatteries {
//				heap.Push(hp, battery-1)
//			}
//		}
//	}

/*
*
结论：
若所剩电池的容量均小于canRun的入参time(碎片电池) 且电池总和大于电脑数*time 则可以满足要求 反之一定无法满足要求
若有电池容量大于time 该电池只供该电脑单独使用 其余电池情况同上
*/
func maxRunTime(n int, batteries []int) int64 {
	var canRun func(time int) bool
	canRun = func(time int) bool {
		computerNum := n
		curBatterySum := 0
		for _, battery := range batteries {
			if battery >= time {
				computerNum--
			} else {
				curBatterySum += battery
			}
		}
		return computerNum <= 0 || (computerNum*time <= curBatterySum)
	}
	batterySum := 0
	for _, battery := range batteries {
		batterySum += battery
	}
	left, right := 0, batterySum/n
	var ans int64 = 0
	for left <= right {
		mid := left + (right-left)/2
		if canRun(mid) {
			ans = int64(mid)
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return ans
}
func smallestDivisor(nums []int, threshold int) int {
	var check func(divisor int) bool
	check = func(divisor int) bool {
		sum := 0
		for _, num := range nums {
			sum += (num + divisor - 1) / divisor
		}
		return sum <= threshold
	}
	maxDivisor := slices.Max(nums)
	return sort.Search(maxDivisor, func(i int) bool {
		return check(i + 1)
	}) + 1
}
func minimumTime(time []int, totalTrips int) int64 {
	var check func(maxTime int) bool
	check = func(maxTime int) bool {
		cnt := 0
		for _, t := range time {
			cnt += maxTime / t
			if cnt >= totalTrips {
				return true
			}
		}
		return cnt >= totalTrips
	}
	maxTime := time[0] * totalTrips
	return int64(sort.Search(maxTime, check))
}

func minimumSize(nums []int, maxOperations int) int {
	var check func(cost int) bool
	check = func(cost int) bool {
		cnt := 0
		for _, num := range nums {
			cnt += (num - 1) / cost
			if cnt > maxOperations {
				return false
			}
		}
		return cnt <= maxOperations
	}
	maxCost := slices.Max(nums)
	return sort.Search(maxCost, func(i int) bool {
		return check(i + 1)
	}) + 1
}
func maximumCandies(candies []int, k int64) int {
	var check func(candy int) bool
	check = func(candy int) bool {
		cnt := 0
		for _, c := range candies {
			cnt += c / candy
			if int64(cnt) >= k {
				return false
			}
		}
		return int64(cnt) < k
	}
	sumCandy := 0
	for _, candy := range candies {
		sumCandy += candy
	}
	maxCandy := slices.Max(candies)
	return sort.Search(maxCandy, func(i int) bool {
		return check(i + 1)
	})
}
func minSpeedOnTime(dist []int, hour float64) int {
	var check func(speed int) bool
	check = func(speed int) bool {
		var time float64 = 0
		for i, d := range dist {
			if i == len(dist)-1 {
				time += float64(d) / float64(speed)
			} else {
				time += math.Ceil(float64(d) / float64(speed))
			}
			if time > hour {
				return false
			}
		}
		return time <= hour
	}
	maxSpeed := 10000000
	ans := sort.Search(maxSpeed, func(i int) bool {
		return check(i + 1)
	}) + 1
	if ans == maxSpeed+1 {
		return -1
	}
	return ans
}
func shipWithinDays(weights []int, days int) int {
	var check func(ship int) bool
	check = func(ship int) bool {
		cnt := 1
		weight := 0
		index := 0
		for index < len(weights) {
			if weight+weights[index] > ship {
				cnt++
				weight = 0
				if cnt > days {
					return false
				}
			} else {
				weight += weights[index]
				index++
			}
		}
		return cnt <= days
	}
	maxShip := 0
	for _, weight := range weights {
		maxShip += weight
	}
	return sort.Search(maxShip, check)
}
func splitWordsBySeparator(words []string, separator byte) (ans []string) {
	for _, word := range words {
		split := strings.Split(word, string(separator))
		for _, s := range split {
			if s != "" {
				ans = append(ans, s)
			}
		}
	}
	return
}
func minimizedMaximum(n int, quantities []int) int {
	var check func(x int) bool
	check = func(x int) bool {
		cnt := 0
		for _, quantity := range quantities {
			cnt += (quantity + x - 1) / x
			if cnt > n {
				return false
			}
		}
		return cnt <= n
	}
	return sort.Search(slices.Max(quantities), func(i int) bool {
		return check(i + 1)
	}) + 1
}

//	func minDays(bloomDay []int, m int, k int) int {
//		var check func(days int) bool
//		check = func(days int) bool {
//			cnt := 0
//			for i := 0; i < len(bloomDay); {
//				end := i
//				for end < len(bloomDay) && bloomDay[end] <= days {
//					end++
//				}
//				cnt += (end - i) / k
//				if cnt >= m {
//					return true
//				}
//				i = end + 1
//			}
//			return cnt >= m
//		}
//		if len(bloomDay) < m*k {
//			return -1
//		}
//		return sort.Search(slices.Max(bloomDay), check)
//	}
func furthestBuilding(heights []int, bricks int, ladders int) int {
	var canReach func(destIndex int) bool
	canReach = func(destIndex int) bool {
		climb := make([]int, 0, len(heights))
		sum := 0
		for i := 0; i < destIndex; i++ {
			dist := heights[i+1] - heights[i]
			if dist > 0 {
				climb = append(climb, dist)
				sum += dist
			}
		}
		if ladders >= len(climb) {
			return true
		}
		slices.Sort(climb)
		for i := 0; i < ladders; i++ {
			sum -= climb[len(climb)-i-1]
		}
		return sum <= bricks
	}
	left, right := 0, len(heights)-1
	ans := 0
	for left <= right {
		mid := left + (right-left)/2
		if canReach(mid) {
			ans = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	return ans
}

// 暴力
func maximumSwap(num int) int {
	res := math.MinInt
	numBytes := []byte(strconv.Itoa(num))
	for i := 0; i < len(numBytes); i++ {
		for j := 0; j < len(numBytes); j++ {
			numBytes[i], numBytes[j] = numBytes[j], numBytes[i]
			val, _ := strconv.Atoi(string(numBytes))
			res = max(res, val)
			numBytes[i], numBytes[j] = numBytes[j], numBytes[i]
		}
	}
	return res
}

// 每日温度：单调栈的最基础用法
func dailyTemperatures(temperatures []int) []int {
	stack := make([]int, 0, len(temperatures))
	ans := make([]int, len(temperatures))
	for i, temperature := range temperatures {
		//相等时候的处理:相等也加入单调栈
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperature {
			ans[stack[len(stack)-1]] = i - stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}

//func sumSubarrayMins(arr []int) int {
//	const MOD = 1000000007
//	n := len(arr)
//	stack := make([]int, 0, n)
//	temp := make([][2]int, n)
//	for i := 0; i < n; i++ {
//		temp[i][0] = -1
//		temp[i][1] = n
//	}
//	for i, num := range arr {
//		for len(stack) > 0 && arr[stack[len(stack)-1]] >= num {
//			if len(stack) > 1 {
//				temp[stack[len(stack)-1]][0] = stack[len(stack)-2]
//			}
//			temp[stack[len(stack)-1]][1] = i
//			stack = stack[:len(stack)-1]
//		}
//		stack = append(stack, i)
//	}
//	for len(stack) > 0 {
//		if len(stack) > 1 {
//			temp[stack[len(stack)-1]][0] = stack[len(stack)-2]
//		}
//		stack = stack[:len(stack)-1]
//	}
//	fmt.Println(temp)
//	ans := 0
//	for i := 0; i < n; i++ {
//		ans += (i - temp[i][0]) * (temp[i][1] - i) * arr[i]
//	}
//	return ans % MOD
//}

// 找到右侧第一个更小的数 维护单调增 相等同样出栈
func finalPrices(prices []int) []int {
	n := len(prices)
	stack := make([]int, 0, n)
	ans := make([]int, n)
	copy(ans, prices)
	for i, price := range prices {
		for len(stack) > 0 && prices[stack[len(stack)-1]] >= price {
			ans[stack[len(stack)-1]] -= prices[i]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}

// 找到左右两侧第一个更小的:能括到哪 相等不出栈
// 无哨兵 需要第二次遍历
func largestRectangleArea(heights []int) int {
	n := len(heights)
	stack := make([]int, 0, n)
	left := make([]int, n)
	right := make([]int, n)
	for i := 0; i < n; i++ {
		left[i] = -1
		right[i] = n
	}
	//单调增
	for i, height := range heights {
		for len(stack) > 0 && heights[stack[len(stack)-1]] > height {
			if len(stack) > 1 {
				left[stack[len(stack)-1]] = stack[len(stack)-2]
			}
			right[stack[len(stack)-1]] = i
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	for len(stack) > 0 {
		if len(stack) > 1 {
			left[stack[len(stack)-1]] = stack[len(stack)-2]
		}
		stack = stack[:len(stack)-1]
	}
	ans := math.MinInt
	for i := 0; i < n; i++ {
		//正方形
		//if right[i]-left[i]-1 >= heights[i] {
		//	ans = max(ans, heights[i]*heights[i])
		//}
		ans = max(ans, (right[i]-left[i]-1)*heights[i])
	}
	return ans
}

// 压缩数组分别以每一行为底 调用 largestRectangleArea
func maximalRectangle(matrix [][]byte) int {
	heights := make([]int, len(matrix[0]))
	ans := math.MinInt
	for _, row := range matrix {
		for i, char := range row {
			if char == '1' {
				heights[i] += 1
			} else {
				heights[i] = 0
			}
		}
		ans = max(ans, largestRectangleArea(heights))
	}
	return ans
}

//单调栈做法
//func maximalSquare(matrix [][]byte) int {
//	heights := make([]int, len(matrix[0]))
//	ans := math.MinInt
//	for _, row := range matrix {
//		for i, char := range row {
//			if char == '1' {
//				heights[i] += 1
//			} else {
//				heights[i] = 0
//			}
//		}
//		ans = max(ans, largestRectangleArea(heights))
//	}
//	return ans
//}

// 暴力
//func maxWidthRamp(nums []int) int {
//	ans := 0
//	for i := 0; i < len(nums); i++ {
//		for j := len(nums) - 1; j-i >= ans; j-- {
//			if nums[j] >= nums[i] {
//				ans = max(ans, j-i)
//			}
//		}
//	}
//	return ans
//}

func removeDuplicateLetters(s string) string {
	count := make(map[byte]int)
	for _, char := range s {
		count[byte(char)]++
	}
	stack := make([]byte, 0, len(count))
	for _, char := range s {
		if !slices.Contains(stack, byte(char)) {
			for len(stack) > 0 && stack[len(stack)-1] >= byte(char) && count[stack[len(stack)-1]] >= 1 {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, byte(char))
		}
		count[byte(char)] -= 1
	}
	return string(stack)
}

// 单调栈的经典用法之外的一个用法：先只入栈 严格维护递增递减栈
// 最远的更大元素
func maxWidthRamp(nums []int) int {
	n := len(nums)
	stack := make([]int, 0, n)
	ans := 0
	//只收集严格单调递减的元素:栈内元素都有作为左边界的可能性
	for i, num := range nums {
		//与当前栈顶元素相等的元素 即使后面存在大于它的元素 与目前的栈顶元素的距离也是更大的
		//大于当前栈顶元素的元素 即使后面存在大于它的元素 它比目前栈顶元素更大 组成的距离也更大
		if len(stack) == 0 || num < nums[stack[len(stack)-1]] {
			stack = append(stack, i)
		}
	}
	//从后向前遍历具有成为右边界可能性的元素
	for i := n - 1; i >= 0 && len(stack) > 0; i-- {
		for len(stack) > 0 && nums[stack[len(stack)-1]] <= nums[i] {
			ans = max(ans, i-stack[len(stack)-1])
			//与当前(最靠右)的元素计算完距离后 就可以弹出
			//因为后续不可能存在更靠右的元素与其计算距离
			stack = stack[:len(stack)-1]
		}
	}
	return ans
}

// 类似上一题 通过前缀和求解和大于0的最长子数组
func longestWPI(hours []int) int {
	prefix := make([]int, len(hours)+1)
	for i := 1; i < len(prefix); i++ {
		prefix[i] = prefix[i-1]
		if hours[i-1] > 8 {
			prefix[i]++
		} else {
			prefix[i]--
		}
	}
	stack := make([]int, 0, len(prefix))
	for i, p := range prefix {
		if len(stack) == 0 || p < prefix[stack[len(stack)-1]] {
			stack = append(stack, i)
		}
	}
	ans := 0
	for i := len(prefix) - 1; i >= 0; i-- {
		for len(stack) > 0 && prefix[i] > prefix[stack[len(stack)-1]] {
			ans = max(ans, i-stack[len(stack)-1])
			stack = stack[:len(stack)-1]
		}
	}
	return ans
}

// 滑动窗口 滑动右指针
func minSubArrayLen(target int, nums []int) int {
	sum := 0
	ans := math.MaxInt
	//每次窗口右边界向右移动一位
	for left, right := 0, 0; right < len(nums); right++ {
		sum += nums[right]
		//查看是否满足条件 窗口左边界可以右移
		for sum >= target {
			//左边界移动过程中收集最小值作为答案
			ans = min(ans, right-left+1)
			sum -= nums[left]
			left++
		}
	}
	if ans == math.MaxInt {
		return 0
	}
	return ans
}

// 简单的滑动窗口
func totalFruit(fruits []int) int {
	count := make(map[int]int)
	ans := math.MinInt
	left := 0
	//遍历右边界
	for right := 0; right < len(fruits); right++ {
		count[fruits[right]]++
		for len(count) > 2 {
			count[fruits[left]]--
			if count[fruits[left]] == 0 {
				delete(count, fruits[left])
			}
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

func alternatingSubarray(nums []int) int {
	ans := -1
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[j]-nums[j-1] == int(math.Pow(-1, float64(j-i+1))) {
				ans = max(ans, j-i+1)
			} else {
				break
			}
		}
	}
	return ans
}

func containsEvery(countS map[byte]int, countT map[byte]int) bool {
	for k, v := range countT {
		if countS[k] < v {
			return false
		}
	}
	return true
}

//func minWindow(s string, t string) string {
//	if len(s) < len(t) {
//		return ""
//	}
//	countS := make(map[byte]int)
//	countT := make(map[byte]int)
//	for _, char := range t {
//		countT[byte(char)]++
//	}
//	ans := ""
//	for left, right := 0, 0; right < len(s); right++ {
//		countS[s[right]]++
//		for left < len(s) && countS[s[left]]-1 >= countT[s[left]] {
//			countS[s[left]]--
//			left++
//		}
//		if (ans == "" || right-left+1 < len(ans)) && containsEvery(countS, countT) {
//			ans = s[left : right+1]
//		}
//	}
//	return ans
//}

func minWindow(s string, t string) string {
	if len(s) < len(t) {
		return ""
	}
	cnt := make(map[byte]int)
	total := len(t)
	for _, char := range t {
		cnt[byte(char)]--
	}
	ans := ""
	for left, right := 0, 0; right < len(s); right++ {
		if cnt[s[right]] < 0 {
			total--
		}
		//每次right向右移动一个
		cnt[s[right]]++
		//保证窗口内至少包含了所有t的字符
		if total == 0 {
			//检查左窗口是否可以移动 子串能否尽量短
			for cnt[s[left]] > 0 {
				cnt[s[left]]--
				left++
			}
			if ans == "" || right-left+1 < len(ans) {
				ans = s[left : right+1]
			}
		}
	}
	return ans
}

// 一种处理环形数组的方法：复制一份连接在原数组后面
func canCompleteCircuit(gas []int, cost []int) int {
	balance := make([]int, len(gas))
	for i := 0; i < len(gas); i++ {
		balance[i] = gas[i] - cost[i]
	}
	balance = append(balance, balance...)
	cnt := 0
	for left, right := 0, 0; right < len(balance); right++ {
		cnt += balance[right]
		for cnt < 0 {
			cnt -= balance[left]
			left++
			if left >= len(gas) {
				return -1
			}
		}
		if right-left+1 == len(gas) {
			return left
		}
	}
	return -1
}

//func balancedString(s string) int {
//	count := make(map[byte]int)
//	for _, char := range s {
//		count[byte(char)]++
//	}
//	sb := strings.Builder{}
//	for k, v := range count {
//		if v > len(s)/4 {
//			for i := 0; i < v-len(s)/4; i++ {
//				sb.WriteByte(k)
//			}
//		}
//	}
//	t := sb.String()
//	if t == "" {
//		return 0
//	}
//	//最小覆盖子串
//	return len(minWindow(s, t))
//}

// 若窗口外的每一个字符数量都小于n/4 则一定能通过转化窗口内的字符使各个字符的数量修正为n/4
// 若窗口外的字符大于n/4 操作窗口内的内容无能为力
func balancedString(s string) int {
	count := make(map[byte]int)
	for _, char := range s {
		count[byte(char)]++
	}
	n := len(s)
	//已经满足条件
	if count['Q'] == n/4 && count['W'] == n/4 && count['E'] == n/4 && count['R'] == n/4 {
		return 0
	}
	ans := n
	//枚举右窗口
	for left, right := 0, 0; right < len(s); right++ {
		count[s[right]]--
		for count['Q'] <= n/4 && count['W'] <= n/4 && count['E'] <= n/4 && count['R'] <= n/4 {
			ans = min(ans, right-left+1)
			count[s[left]]++
			left++
		}
	}
	return ans
}

// 小技巧：恰好k=小于等于k=>小于等于k-1
// 窗口括的越大 种类小于等于k的可能性越低
// 即一旦种类大于k 必须从左边界开始缩小范围 才有可能让种类数重新小于等于k
// "恰好"这个概念没有单调性
func subarraysWithKDistinct(nums []int, k int) int {
	var count func(m int) int
	count = func(m int) int {
		ans := 0
		cnt := make(map[int]int)
		for left, right := 0, 0; right < len(nums); right++ {
			cnt[nums[right]]++
			for len(cnt) > m {
				cnt[nums[left]]--
				if cnt[nums[left]] == 0 {
					delete(cnt, nums[left])
				}
				left++
			}
			ans += right - left + 1
		}
		return ans
	}
	return count(k) - count(k-1)
}

// 规定只能含有n个字符 遍历n从0到26 求最大值
// 敢于模拟 通过模拟的过程确定需要什么维护什么信息

func numSubarrayProductLessThanK(nums []int, k int) int {
	ans := 0
	product := 1
	for left, right := 0, 0; right < len(nums); right++ {
		product *= nums[right]
		for left <= right && product >= k {
			product /= nums[left]
			left++
		}
		ans += right - left + 1
	}
	return ans
}

// 暴力
//func maximumSumOfHeights(maxHeights []int) int64 {
//	var ans int64 = math.MinInt
//	for i := 0; i < len(maxHeights); i++ {
//		heights := make([]int, len(maxHeights))
//		heights[i] = maxHeights[i]
//		sum := heights[i]
//		for j := i - 1; j >= 0; j-- {
//			heights[j] = min(maxHeights[j], heights[j+1])
//			sum += heights[j]
//		}
//		for j := i + 1; j < len(maxHeights); j++ {
//			heights[j] = min(maxHeights[j], heights[j-1])
//			sum += heights[j]
//		}
//		ans = max(ans, int64(sum))
//	}
//	return ans
//}

// 已经是线性复杂度了还是超时 还可以再优化
//func maximumSumOfHeights(maxHeights []int) int64 {
//	pre := make([]int, len(maxHeights))
//	stack := make([]int, 0, len(maxHeights))
//	sum := 0
//	for i := 0; i < len(maxHeights); i++ {
//		j := len(stack) - 1
//		这里的循环时间复杂度太高 信息可以压缩
//		for j >= 0 && stack[j] > maxHeights[i] {
//			sum -= stack[j] - maxHeights[i]
//			stack[j] = maxHeights[i]
//			j--
//		}
//		stack = append(stack, maxHeights[i])
//		sum += maxHeights[i]
//		pre[i] = sum
//	}
//	sum = 0
//	stack = make([]int, 0, len(maxHeights))
//	ans := 0
//	for i := len(maxHeights) - 1; i >= 0; i-- {
//		j := len(stack) - 1
//		for j >= 0 && stack[j] > maxHeights[i] {
//			sum -= stack[j] - maxHeights[i]
//			stack[j] = maxHeights[i]
//			j--
//		}
//		stack = append(stack, maxHeights[i])
//		sum += maxHeights[i]
//		ans = max(ans, pre[i]+sum-maxHeights[i])
//	}
//	return int64(ans)
//}

// 山峰两侧独立 前后缀分解+单调栈 参考灵神思路
func maximumSumOfHeights(maxHeights []int) int64 {
	n := len(maxHeights)
	ans := 0
	stack := make([]int, 0, n)
	sum := 0
	pre := make([]int, n)
	//正序 维护从栈底到栈顶单调增的单调栈
	for i, h := range maxHeights {
		for len(stack) > 0 && maxHeights[stack[len(stack)-1]] > h {
			//没有用哨兵 需要对栈是否为空进行特判
			if len(stack) > 1 {
				sum -= (stack[len(stack)-1] - stack[len(stack)-2]) * (maxHeights[stack[len(stack)-1]] - h)
			} else {
				sum -= (stack[len(stack)-1] + 1) * (maxHeights[stack[len(stack)-1]] - h)
			}
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
		sum += h
		pre[i] = sum
	}
	stack = make([]int, 0, n)
	sum = 0
	//倒序
	for i := n - 1; i >= 0; i-- {
		h := maxHeights[i]
		for len(stack) > 0 && maxHeights[stack[len(stack)-1]] > h {
			if len(stack) > 1 {
				sum -= (stack[len(stack)-2] - stack[len(stack)-1]) * (maxHeights[stack[len(stack)-1]] - h)
			} else {
				sum -= (n - stack[len(stack)-1]) * (maxHeights[stack[len(stack)-1]] - h)
			}
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
		//随时结算答案 不需要另外维护suf
		sum += h
		//以i为山峰：左侧(包括当前)+右侧(包括当前)-当前
		ans = max(ans, pre[i]+sum-h)
	}
	return int64(ans)
}

func largestNumber1(nums []int) string {
	//自定义排序：a拼b和b拼a 谁更大谁就放在前
	//而不是直接字典序排序
	slices.SortFunc(nums, func(a, b int) int {
		strA := strconv.Itoa(a)
		strB := strconv.Itoa(b)
		return strings.Compare(strB+strA, strA+strB)
	})
	sb := strings.Builder{}
	for _, num := range nums {
		sb.WriteString(strconv.Itoa(num))
	}
	ans := sb.String()
	if ans[0] == '0' {
		return "0"
	}
	return ans
}

// 回溯
func largestNumber2(nums []int) string {
	res := make([]string, 0)
	used := make([]bool, len(nums))
	var dfs func(i int, path string)
	dfs = func(i int, path string) {
		if i == len(nums) {
			res = append(res, path)
			return
		}
		for index, num := range nums {
			if !used[index] {
				used[index] = true
				dfs(i+1, path+strconv.Itoa(num))
				used[index] = false
			}
		}
	}
	dfs(0, "")
	slices.SortFunc(res, func(a, b string) int {
		return strings.Compare(b, a)
	})
	return res[0]
}
func twoCitySchedCost(costs [][]int) int {
	slices.SortFunc(costs, func(a, b []int) int {
		return (a[0] - a[1]) - (b[0] - b[1])
	})
	n := len(costs)
	ans := 0
	for i := 0; i < n; i++ {
		if i < n/2 {
			ans += costs[i][0]
		} else {
			ans += costs[i][1]
		}
	}
	return ans
}

// 回溯
func twoCitySchedCost2(costs [][]int) int {
	res := make([]int, 0)
	n := len(costs)
	var dfs func(aCount, cost, i int)
	dfs = func(aCount, cost, i int) {
		if i == n {
			if aCount == n/2 {
				res = append(res, cost)
			}
			return
		}
		if aCount < n/2 {
			dfs(aCount+1, cost+costs[i][0], i+1)
			dfs(aCount, cost+costs[i][1], i+1)
		} else {
			dfs(aCount, cost+costs[i][1], i+1)
		}
	}
	dfs(0, 0, 0)
	slices.Sort(res)
	return res[0]
}
