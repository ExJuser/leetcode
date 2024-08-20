package common

import (
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
)

// 2684. 矩阵中移动的最大次数 普通二维dp
func maxMoves(grid [][]int) int {
	dp := make([][]int, len(grid))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(grid[i]))
	}
	var ans int
	for j := 0; j < len(dp[0]); j++ {
		for i := 0; i < len(grid); i++ {
			if dp[i][j] == j { //可以到达当前格子
				//尝试三个方向
				if j+1 < len(grid[0]) {
					if dp[i][j+1] != j+1 {
						if grid[i][j+1] > grid[i][j] {
							dp[i][j+1] = j + 1
							ans = max(ans, dp[i][j+1])
						}
					}
					if i-1 >= 0 && dp[i-1][j+1] != j+1 {
						if grid[i-1][j+1] > grid[i][j] {
							dp[i-1][j+1] = j + 1
							ans = max(ans, dp[i-1][j+1])
						}
					}
					if i+1 < len(grid) && dp[i+1][j+1] != j+1 {
						if grid[i+1][j+1] > grid[i][j] {
							dp[i+1][j+1] = j + 1
							ans = max(ans, dp[i+1][j+1])
						}
					}
				}
			}
		}
	}
	return ans
}

// 464. 我能赢吗 博弈 状压dp
func canIWin(maxChoosableInteger int, desiredTotal int) bool {
	if desiredTotal == 0 {
		return true
	}
	if (maxChoosableInteger*(1+maxChoosableInteger))/2 < desiredTotal {
		return false
	}
	//dfs 当前可以选择的数字 还差多少满足累计和 返回能否稳赢
	//可以选择的数字使用状态位表示 挂一个缓存表
	state := (1 << (maxChoosableInteger + 1)) - 2
	cache := make(map[int]bool)
	var dfs func(state int, desired int) bool
	dfs = func(state int, desired int) bool {
		if desired <= 0 {
			return false
		}
		if v, ok := cache[state]; ok {
			return v
		}
		for i := 1; i <= maxChoosableInteger; i++ {
			//这个数字不能被使用过
			if state&(1<<i) != 0 {
				//对手不能赢
				if !dfs(state^(1<<i), desired-i) {
					cache[state] = true
					return true
				} else {
					cache[state^(1<<i)] = true
				}
			}
		}
		cache[state] = false
		return false
	}
	return dfs(state, desiredTotal)
}

// 698. 划分为k个相等的子集 状压dp+缓存表
func canPartitionKSubsets(nums []int, k int) bool {
	n := len(nums)
	var sum int
	for _, num := range nums {
		sum += num
	}
	if sum%k != 0 {
		return false
	}
	limit := sum / k
	//所有元素都可以使用的状态
	state := (1 << n) - 1
	//缓存每个状态的结果
	cache := make([]int, 1<<n)
	var dfs func(state, cur, rest int, cache []int) bool
	dfs = func(state, cur, rest int, cache []int) bool {
		if rest == 0 {
			return true
		}
		if cache[state] != 0 { //之前计算过
			return cache[state] == 1
		}
		var ans bool
		for i := 0; i < n; i++ {
			//还没被用过而且不超出范围
			if state&(1<<i) != 0 && cur+nums[i] <= limit {
				if cur+nums[i] == limit {
					ans = dfs(state^(1<<i), 0, rest-1, cache)
				} else {
					ans = dfs(state^(1<<i), cur+nums[i], rest, cache)
				}
				if ans {
					break
				}
			}
		}
		if ans {
			cache[state] = 1
		} else {
			cache[state] = -1
		}
		return ans
	}
	return dfs(state, 0, k, cache)
}

// 698.火柴拼正方形 状压dp+缓存表
// 缓存表的用法 在dfs中作为参数携带
// 先判断base case 再判断是否有缓存 返回前设置缓存的值
func makesquare(matchsticks []int) bool {
	n := len(matchsticks)
	var sum int
	for _, num := range matchsticks {
		sum += num
	}
	if sum%4 != 0 {
		return false
	}
	limit := sum / 4
	//所有元素都可以使用的状态
	state := (1 << n) - 1
	//缓存每个状态的结果
	cache := make([]int, 1<<n)
	var dfs func(state, cur, rest int, cache []int) bool
	dfs = func(state, cur, rest int, cache []int) bool {
		if rest == 0 {
			return true
		}
		if cache[state] != 0 { //之前计算过
			return cache[state] == 1
		}
		var ans bool
		for i := 0; i < n; i++ {
			//还没被用过而且不超出范围
			if state&(1<<i) != 0 && cur+matchsticks[i] <= limit {
				if cur+matchsticks[i] == limit {
					ans = dfs(state^(1<<i), 0, rest-1, cache)
				} else {
					ans = dfs(state^(1<<i), cur+matchsticks[i], rest, cache)
				}
				if ans {
					break
				}
			}
		}
		if ans {
			cache[state] = 1
		} else {
			cache[state] = -1
		}
		return ans
	}
	return dfs(state, 0, 4, cache)
}

// 最长相同子序列 初始化比较麻烦
func maxUncrossedLines(nums1 []int, nums2 []int) int {
	dp := make([][]int, len(nums1))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(nums2))
	}
	for i := 0; i < len(dp); i++ {
		if nums1[i] == nums2[0] {
			for ; i < len(dp); i++ {
				dp[i][0] = 1
			}
			break
		}
	}
	for i := 0; i < len(dp[0]); i++ {
		if nums2[i] == nums1[0] {
			for ; i < len(dp[0]); i++ {
				dp[0][i] = 1
			}
			break
		}
	}
	for i := 1; i < len(nums1); i++ {
		for j := 1; j < len(nums2); j++ {
			if nums1[i] == nums2[j] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[len(nums1)-1][len(nums2)-1]
}

// 179. 最大数
func largestNumber(nums []int) string {
	slices.SortFunc(nums, func(a, b int) int {
		strA := strconv.Itoa(a)
		strB := strconv.Itoa(b)
		return strings.Compare(strB+strA, strA+strB)
	})
	sb := strings.Builder{}
	for _, num := range nums {
		sb.WriteString(strconv.Itoa(num))
	}
	str := sb.String()
	//去掉前导0 找到第一个非零索引
	var i int
	if str[0] == '0' && len(str) > 1 {
		for i < len(str) && str[i] == '0' {
			i++
		}
	}
	if i == len(str) {
		return "0"
	}
	return str[i:]
}

// 快速选择的多数元素
func majorityElement(nums []int) int {
	var helper func(left, right, k int) int
	helper = func(left, right, k int) int {
		if left >= right {
			return nums[k]
		}
		i, j := left, right
		pivot := nums[rand.IntN(right-left+1)+left]
		for i <= j {
			for nums[i] < pivot {
				i++
			}
			for nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		if j >= k {
			return helper(left, j, k)
		} else {
			return helper(i, right, k)
		}
	}
	return helper(0, len(nums)-1, len(nums)/2)
}

func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		dp[i][0] = 1
	}
	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

// 1143. 最长公共子序列
func longestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)
	for i := 0; i < m+1; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if text1[i] == text2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j], dp[i][j+1], dp[i+1][j])
			}
		}
	}
	return dp[m][n]
}

// 72. 编辑距离 多申请一位
func minDistance(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)
	for i := 0; i < m+1; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i < m+1; i++ {
		dp[i][0] = i
	}
	for i := 1; i < n+1; i++ {
		dp[0][i] = i
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if word1[i] == word2[j] {
				dp[i+1][j+1] = dp[i][j]
			} else {
				dp[i+1][j+1] = min(dp[i][j+1]+1, dp[i+1][j]+1, dp[i][j]+1)
			}
		}
	}
	return dp[m][n]
}
func numSquares(n int) int {
	dp := make([]int, n+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = math.MaxInt
	}
	for i := 0; i*i <= n; i++ {
		dp[i*i] = 1
	}
	for i := 1; i <= n; i++ {
		for j := 1; i-j*j >= 0; j++ {
			dp[i] = min(dp[i], dp[i-j*j]+1)
		}
	}

	return dp[n]
}

// 322. 零钱兑换 完全背包
func coinChange(coins []int, amount int) int {
	//凑成amount的最少coin数
	//dpi 装满i容量的背包所需的最少coin数
	dp := make([]int, amount+1)
	for i := 1; i < len(dp); i++ {
		dp[i] = math.MaxInt
	}
	for i := 0; i <= amount; i++ {
		for j := 0; j < len(coins); j++ {
			if i-coins[j] >= 0 && dp[i-coins[j]] != math.MaxInt {
				dp[i] = min(dp[i], dp[i-coins[j]]+1)
			}
		}
	}
	if dp[amount] == math.MaxInt {
		return -1
	}
	return dp[amount]
}
func maxProfit2(prices []int) int {
	dp := make([][2]int, len(prices))
	dp[0][0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[len(prices)-1][1]
}

// 274. H 指数
func hIndex(citations []int) int {
	//h指数为2代表至少有两篇论文的引用次数大于2
	var check func(hValue int) bool
	check = func(hValue int) bool {
		var cnt int
		for _, citation := range citations {
			if citation >= hValue {
				cnt++
			}
		}
		return cnt >= hValue
	}
	left, right := 0, 1000
	var ans int
	for left <= right {
		mid := (right-left)/2 + left
		if check(mid) {
			left = mid + 1
			ans = mid
		} else {
			right = mid - 1
		}
	}
	return ans
}

type RandomizedSet struct {
	mp    map[int]int
	array []int
}

//func Constructor() RandomizedSet {
//	return RandomizedSet{mp: make(map[int]int), array: make([]int, 0)}
//}

func (this *RandomizedSet) Insert(val int) bool {
	if _, ok := this.mp[val]; !ok { //不存在
		this.mp[val] = len(this.array)
		this.array = append(this.array, val)
		return true
	}
	return false
}

func (this *RandomizedSet) Remove(val int) bool {
	if index, ok := this.mp[val]; ok {
		//要删除的元素的索引为index
		//将他和数组的最后一个位置交换 更改最后一个位置元素的索引
		last := len(this.array) - 1
		this.mp[this.array[last]] = index
		this.array[index], this.array[last] = this.array[last], this.array[index]
		this.array = this.array[:len(this.array)-1]
		delete(this.mp, val)
		return true
	}
	return false
}

func (this *RandomizedSet) GetRandom() int {
	return this.array[rand.IntN(len(this.array))]
}

func MergeSlice(nums1 []int, nums2 []int) []int {
	newNums := make([]int, 0, len(nums1)+len(nums2))
	var i, j, index int
	for i < len(nums1) && j < len(nums2) {
		if nums1[i] <= nums2[j] {
			newNums = append(newNums, nums1[i])
			i++
		} else {
			newNums = append(newNums, nums2[j])
			j++
		}
		index++
	}
	if i < len(nums1) {
		newNums = append(newNums, nums1[i:]...)
	}
	if j < len(nums2) {
		newNums = append(newNums, nums2[j:]...)
	}
	return newNums
}

// 395. 至少有 K 个重复字符的最长子串
func longestSubstring(s string, k int) int {
	//每一个字符的出现次数都不少于k
	//先遍历得到每个字符的出现频次 如果字符总出现次数小于k 则子串不能包含它 分治分为前和后
	mp := make(map[byte]int, 26)
	for _, ch := range s {
		mp[byte(ch)]++
	}
	for i := 0; i < len(s); i++ {
		if mp[s[i]] < k {
			return max(longestSubstring(s[0:i], k), longestSubstring(s[i+1:], k))
		}
	}
	return len(s)
}

// 93. 复原 IP 地址
func restoreIpAddresses(s string) (ans []string) {
	//选择切割还是不切割
	//先写一个判断切割出来的是否合法的函数
	var check func(i, j int) bool
	check = func(i, j int) bool {
		if i > j {
			return false
		}
		ip := s[i : j+1]
		//不能含有前导0 如果第一位为0 那他只能是0
		if ip[0] == '0' {
			return len(ip) == 1
		}
		num, _ := strconv.Atoi(ip)
		return num > 0 && num <= 255
	}
	var dfs func(i, j int, path []string)
	dfs = func(i, j int, path []string) {
		if j == len(s) {
			if len(path) == 4 {
				ans = append(ans, strings.Join(path, "."))
			}
			return
		}
		//选择切割还是不切割
		//切割首先需要满足的条件
		if len(path) < 4 && check(i, j) {
			path = append(path, s[i:j+1])
			dfs(j+1, j+1, path)
			path = path[:len(path)-1]
		}
		//不切割
		if j < len(s)-1 { //如果到了最后一位必须切割
			dfs(i, j+1, path)
		}
	}
	dfs(0, 0, []string{})
	return
}

// 一个都不留
//func deleteDuplicates(head *ListNode) *ListNode {
//	//假设这里的dfs已经完成了一个节点和其后续节点的去重操作
//	var dfs func(node *ListNode) *ListNode
//	dfs = func(node *ListNode) *ListNode {
//		if node == nil || node.Next == nil {
//			return node
//		}
//		//如果当前节点和后续节点不同 需要保留当前节点 再清理后续节点
//		if node.Val != node.Next.Val {
//			node.Next = dfs(node.Next)
//			return node
//		} else {
//			val := node.Val
//			for node != nil && node.Val == val {
//				node = node.Next
//			}
//			return dfs(node)
//		}
//	}
//	return dfs(head)
//}

// 从右向左 找到第一个左小于右的位置  再次从右开始找到第一个大于这个数的
func nextPermutation(nums []int) {
	if len(nums) <= 1 {
		return
	}
	i, j, k := len(nums)-2, len(nums)-1, len(nums)-1
	for i >= 0 && nums[i] >= nums[j] {
		i--
		j--
	}
	if i >= 0 {
		for nums[i] >= nums[k] {
			k--
		}
		nums[i], nums[k] = nums[k], nums[i]
	}
	slices.Reverse(nums[j:])
}
func compareVersion(version1 string, version2 string) int {
	ver1 := strings.Split(version1, ".")
	ver2 := strings.Split(version2, ".")
	i := 0
	for ; i < len(ver1) && i < len(ver2); i++ {
		v1, _ := strconv.Atoi(ver1[i])
		v2, _ := strconv.Atoi(ver2[i])
		if v1 < v2 {
			return -1
		} else if v1 > v2 {
			return 1
		}
	}
	if i < len(ver1) { //判断后面是否全为0 如果全为0 返回0
		for ; i < len(ver1); i++ {
			v, _ := strconv.Atoi(ver1[i])
			if v > 0 {
				return 1
			}
		}
	} else {
		for ; i < len(ver2); i++ {
			v, _ := strconv.Atoi(ver2[i])
			if v > 0 {
				return -1
			}
		}
	}
	return 0
}

func firstMissingPositive(nums []int) int {
	for i := 0; i < len(nums); i++ {
		for nums[i] > 0 && nums[i] < len(nums) && nums[nums[i]-1] != nums[i] {
			nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
		}
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return len(nums) + 1
}

// 先用栈模拟一遍找到不合法的括号位置
func longestValidParentheses(s string) int {
	stack := make([]int, 0, len(s))
	for i, ch := range s {
		if ch == '(' {
			stack = append(stack, i)
		} else {
			if len(stack) > 0 && s[stack[len(stack)-1]] == '(' {
				stack = stack[:len(stack)-1]
			} else {
				stack = append(stack, i)
			}
		}
	}
	//遍历一遍得到不合法的位置
	sequence := make([]int, len(s))
	for i := 0; i < len(stack); i++ {
		sequence[stack[i]] = 1
	}
	//寻找最长的i序列
	var ans int
	var length int
	for i := 0; i < len(sequence); i++ {
		if sequence[i] == 0 {
			length++
			ans = max(ans, length)
		} else {
			length = 0
		}
	}
	return ans
}

func reverseWords(s string) string {
	//遇到空格就向后找到第一个非空格
	//遇到非空格就向后找到第一个空格
	strs := make([]string, 0, len(s))
	var i int
	for i < len(s) {
		if s[i] == ' ' {
			for i < len(s) && s[i] == ' ' {
				i++
			} //出来之后要么是非空格要么结束
		} else { //非空格
			//向后找到第一个空格
			j := i
			for j < len(s) && s[j] != ' ' {
				j++
			}
			//出来之后j要么到了末尾 要么是空格
			strs = append(strs, s[i:j])
			i = j + 1
		}
	}
	slices.Reverse(strs)
	return strings.Join(strs, " ")
}

// 129. 求根节点到叶节点数字之和
func sumNumbers(root *TreeNode) (ans int) {
	var dfs func(node *TreeNode, sum int)
	dfs = func(node *TreeNode, sum int) {
		if node == nil {
			return
		}
		if node.Left == nil && node.Right == nil {
			ans += 10*sum + node.Val
			return
		}
		sum = sum*10 + node.Val
		dfs(node.Left, sum)
		dfs(node.Right, sum)
	}
	dfs(root, 0)
	return
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
//
//func isSymmetric(root *TreeNode) bool {
//	return isSameTree(root.Left, root.Right)
//}
