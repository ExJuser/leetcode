package main

import (
	"math"
	"slices"
	"sort"
	"strings"
)

func mincostTickets(days []int, costs []int) int {
	//dp数组含义：完成这一天及之后的旅行的所有花费
	dp := make([]int, len(days)+1)
	duration := []int{1, 7, 30}
	//遍历顺序：从后向前
	for i := len(days) - 1; i >= 0; i-- {
		ans := math.MaxInt
		for j, cost := range costs {
			k := i
			for k < len(days) && days[i]+duration[j] > days[k] {
				k++
			}
			ans = min(ans, cost+dp[k])
		}
		dp[i] = ans
	}
	return dp[0]
}

// 回溯会超时
//func coinChange(coins []int, amount int) int {
//	ans := math.MaxInt
//	var dfs func(index, curAmount, count int)
//	dfs = func(index, curAmount, count int) {
//		//如果当前硬币数量已经超过了目前的最小数量
//		////如果总金额超过了目标金额
//		if count > ans || curAmount > amount {
//			return
//		}
//		if index == len(coins) {
//			if curAmount == amount {
//				ans = min(ans, count)
//			}
//			return
//		}
//		//选
//		dfs(index, curAmount+coins[index], count+1)
//		//不选
//		dfs(index+1, curAmount, count)
//	}
//	dfs(0, 0, 0)
//	if ans == math.MaxInt {
//		return -1
//	}
//	return ans
//}

//func numDecodings(s string) int {
//	dp := make([]int, len(s)+1)
//	dp[len(s)] = 1
//	for i := len(s) - 1; i >= 0; i-- {
//		if s[i] == '0' {
//			dp[i] = 0
//		} else {
//			dp[i] = dp[i+1]
//			if i+1 < len(s) && (s[i]-'0')*10+s[i+1]-'0' <= 26 {
//				dp[i] += dp[i+2]
//			}
//		}
//	}
//	return dp[0]
//}

func numDecodings(s string) int {
	dp := make([]int, len(s)+1)
	dp[0] = 1
	for i := 0; i < len(s); i++ {
		if s[i] != '0' {
			dp[i+1] = dp[i]
		}
		if i-1 >= 0 && s[i-1] != '0' && (s[i-1]-'0')*10+s[i]-'0' <= 26 {
			dp[i+1] += dp[i-1]
		}
	}
	return dp[len(s)]
}

//	func coinChange(coins []int, amount int) int {
//		dp := make([]int, amount+1)
//		dp[0] = 0
//		for j := 1; j <= amount; j++ {
//			dp[j] = math.MaxInt
//			for _, coin := range coins {
//				if j >= coin && dp[j-coin] != math.MaxInt {
//					//因为需要取最小 所以初始化为最大防止被覆盖
//					dp[j] = min(dp[j], dp[j-coin]+1)
//				}
//			}
//		}
//		if dp[amount] == math.MaxInt {
//			return -1
//		}
//		return dp[amount]
//	}

// 和组合还是排列没有关系 因此遍历顺序两种都可以
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0
	for j := 1; j <= amount; j++ {
		dp[j] = math.MaxInt
	}
	for _, coin := range coins {
		for j := coin; j <= amount; j++ {
			if dp[j-coin] != math.MaxInt {
				dp[j] = min(dp[j], dp[j-coin]+1)
			}
		}
	}
	if dp[amount] == math.MaxInt {
		return -1
	}
	return dp[amount]
}

// 反转结束后：
// cur指向的是下一个节点(在反转全部链表节点中即空节点)
// pre指向的是新链表的头结点
//func reverseList(head *ListNode) *ListNode {
//	var pre *ListNode
//	cur := head
//	for cur != nil {
//		nxt := cur.Next
//		cur.Next = pre
//		pre, cur = cur, nxt
//	}
//	return pre
//}

// 有可能对头结点进行操作 需要添加dummy节点
//func reverseBetween(head *ListNode, left int, right int) *ListNode {
//	dummy := &ListNode{Next: head}
//	p0 := dummy
//	//找到要被反转的链表头结点的前一个节点
//	for i := 1; i < left; i++ {
//		p0 = p0.Next
//	}
//	var pre *ListNode
//	cur := p0.Next
//	for i := 0; i < right-left+1; i++ {
//		//反转链表的核心逻辑就这三行
//		nxt := cur.Next
//		cur.Next = pre
//		pre, cur = cur, nxt
//	}
//	p0.Next.Next = cur
//	p0.Next = pre
//	return dummy.Next
//}
//func reverseKGroup(head *ListNode, k int) *ListNode {
//	cnt := 0
//	for p := head; p != nil; p = p.Next {
//		cnt++
//	}
//	dummy := &ListNode{Next: head}
//	p0 := dummy
//	cur := p0.Next
//	for i := 0; i < cnt/k; i++ {
//		var pre *ListNode
//		//翻转链表的核心逻辑仍然是这三行
//		for i := 0; i < k; i++ {
//			nxt := cur.Next
//			cur.Next = pre
//			pre, cur = cur, nxt
//		}
//		//当不是反转整个链表时就需要这两行
//		p1 := p0.Next
//		p0.Next.Next = cur
//		p0.Next = pre
//		p0 = p1
//		//for i := 0; i < k; i++ {
//		//	p0 = p0.Next
//		//}
//	}
//	return dummy.Next
//}

// 01背包：能否装满背包
func canPartition(nums []int) bool {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if sum%2 != 0 {
		return false
	}
	target := sum / 2
	dp := make([]int, target+1)
	for _, num := range nums {
		for j := target; j >= num; j-- {
			dp[j] = max(dp[j], dp[j-num]+num)
		}
	}
	return dp[target] == target
}

/*
*
尽量装满背包
难点在于如何看出是01背包问题
01背包：从序列中选出子序列，使其最接近target
将石头分成尽可能均等的两堆互相撞得到的就是最小值
*/
//func lastStoneWeightII(stones []int) int {
//	sum := 0
//	for _, stone := range stones {
//		sum += stone
//	}
//	target := sum / 2
//	dp := make([]int, target+1)
//	for _, stone := range stones {
//		for j := target; j >= stone; j-- {
//			dp[j] = max(dp[j], dp[j-stone]+stone)
//		}
//	}
//	return Abs(sum - 2*dp[target])
//}

/*
*
有多少种装满背包的方法：dp[j] += dp[j-num] 范式 记住
分成两个集合 一个集合加号 一个集合减号
所以有等式：plus+minus=sum plus-minus=target => plus=(sum+target)/2 minus=(sum-target)/2
*/
func findTargetSumWays(nums []int, target int) int {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	if sum+target < 0 || (sum+target)%2 != 0 {
		return 0
	}
	//转化为装满容量为(sum+target)/2的背包有多少种方法
	//dp[j] 装满容量为j的背包有多少种方法
	dp := make([]int, (sum+target)/2+1)
	//记住：装满容量为0的背包有1种方法
	dp[0] = 1
	for _, num := range nums {
		for j := (sum + target) / 2; j >= num; j-- {
			//对于num这个物品 装满j容量的背包必须由装满背包的方法得来
			//因此有多少种装满j-num的方法 就有多少种装满j的方法 而不是加1
			//累加：装满j可以先装0 再装j；先装1再装j-1；如果存在
			dp[j] += dp[j-num]
		}
	}
	return dp[(sum+target)/2]
}
func onesCount(str string) (cnt int) {
	for _, char := range str {
		if char == '1' {
			cnt++
		}
	}
	return
}

func findMaxForm(strs []string, m int, n int) int {
	count := make([][2]int, len(strs))
	for i, str := range strs {
		cnt := onesCount(str)
		count[i] = [2]int{len(str) - cnt, cnt}
	}
	dp := make([][]int, 0, m+1)
	for i := 0; i < m+1; i++ {
		dp = append(dp, make([]int, n+1))
	}
	for i, _ := range strs {
		for j1 := m; j1 >= count[i][0]; j1-- {
			for j2 := n; j2 >= count[i][1]; j2-- {
				dp[j1][j2] = max(dp[j1][j2], dp[j1-count[i][0]][j2-count[i][1]]+1)
			}
		}
	}
	return dp[m][n]
}

// 完全背包的组合问题：顺序无所谓
// 先遍历物品再遍历背包 都是从小到大 之所以可以正序遍历是因为物品有无限个
func change(amount int, coins []int) int {
	dp := make([]int, amount+1)
	dp[0] = 1
	for _, coin := range coins {
		for j := coin; j <= amount; j++ {
			dp[j] += dp[j-coin]
		}
	}
	return dp[amount]
}

// 完全背包的排列问题：顺序不同才算
// 先遍历背包再遍历物品 都是从小到大
// 爬楼梯其实就是完全背包问题：dp[i]=dp[i-1]+dp[i-2]+...
func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for j := 0; j <= target; j++ {
		for _, num := range nums {
			if j >= num {
				dp[j] += dp[j-num]
			}
		}
	}
	return dp[target]
}

func digitSumString(num string) (sum int) {
	for _, char := range num {
		sum += int(char - '0')
	}
	return
}
func addNums(num1, num2 string) string {
	builder := strings.Builder{}
	i := len(num1) - 1
	j := len(num2) - 1
	var carry byte
	for i >= 0 || j >= 0 {
		var val1, val2 byte
		if i >= 0 {
			val1 = num1[i] - '0'
		}
		if j >= 0 {
			val2 = num2[j] - '0'
		}
		val := carry + val1 + val2
		carry = val / 10
		builder.WriteByte(val%10 + '0')
		if i >= 0 {
			i--
		}
		if j >= 0 {
			j--
		}
	}
	if carry > 0 {
		builder.WriteByte(1 + '0')
	}
	return reverseString(builder.String())
}
func reverseString(s string) string {
	byteArr := []byte(s)
	slices.Reverse(byteArr)
	return string(byteArr)
}
func compareString(str1, str2 string) int {
	if len(str1) != len(str2) {
		return len(str1) - len(str2)
	} else {
		return strings.Compare(str1, str2)
	}
}
func count(num1 string, num2 string, min_sum int, max_sum int) (cnt int) {
	num := num1
	for compareString(num, num2) <= 0 {
		sum := digitSumString(num)
		if sum <= max_sum && sum >= min_sum {
			cnt = (cnt + 1) % (1e9 + 7)
		}
		num = addNums(num, "1")
	}
	return
}

/**
01背包：物品数量有限 先遍历物品再倒序遍历背包
完全背包：物品数量无限 组合问题先遍历物品再正序遍历背包 排列问题先正序遍历背包再正序遍历物品
*/

func numSquares(n int) int {
	squares := make([]int, 0, 31)
	for i := 1; i <= int(math.Sqrt(float64(n))); i++ {
		squares = append(squares, i*i)
	}
	dp := make([]int, n+1)
	for j := 1; j <= n; j++ {
		dp[j] = math.MaxInt
		for _, square := range squares {
			if j >= square && dp[j-square] != math.MaxInt {
				dp[j] = min(dp[j], dp[j-square]+1)
			}
		}
	}
	return dp[n]
}

// 能不能用回溯来做 : 超时
//
//	func wordBreak(s string, wordDict []string) bool {
//		var dfs func(str string) bool
//		dfs = func(str string) bool {
//			if str == "" {
//				return true
//			}
//			flag := false
//			for _, word := range wordDict {
//				if strings.HasPrefix(str, word) {
//					flag = true
//					break
//				}
//			}
//			if flag {
//				for _, word := range wordDict {
//					if strings.HasPrefix(str, word) && dfs(str[len(word):]) {
//						return true
//					}
//				}
//			}
//			return false
//		}
//		return dfs(s)
//	}
//

// 字符串：背包
// 字典：物品
// 可以使用多次：完全背包问题
// 对顺序有要求：排列问题
// 先遍历背包
func wordBreak(s string, wordDict []string) bool {
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for j := 1; j <= len(s); j++ {
		for _, word := range wordDict {
			if j >= len(word) {
				dp[j] = dp[j] || (dp[j-len(word)] && s[j-len(word):j] == word)
			}
		}
	}
	return dp[len(s)]
}

//func rob(nums []int) int {
//	n := len(nums)
//	if n == 1 {
//		return nums[0]
//	}
//	if n == 2 {
//		return max(nums[0], nums[1])
//	}
//	dp := make([][2]int, len(nums))
//	dp[0][0] = nums[0]
//	dp[0][1] = 0
//	dp[1][0] = nums[0]
//	dp[1][1] = nums[1]
//	for i := 2; i < len(nums); i++ {
//		if i == n-1 {
//			return max(dp[i-1][0], dp[i-2][1]+nums[i], dp[i-1][1])
//		}
//		dp[i][0] = max(dp[i-1][0], dp[i-2][0]+nums[i])
//		dp[i][1] = max(dp[i-1][1], dp[i-2][1]+nums[i])
//	}
//	return -1
//}

// 树形dp
// dp[0]不偷当前节点的最大金钱
// dp[1]偷当前节点的最大金钱
// 后序遍历 return 根节点的状态 max(dp[0],dp[1])
//func rob(root *TreeNode) int {
//	var dfs func(node *TreeNode) [2]int
//	dfs = func(node *TreeNode) [2]int {
//		if node == nil {
//			return [2]int{0, 0}
//		}
//		left := dfs(node.Left)
//		right := dfs(node.Right)
//		return [2]int{max(left[0], left[1]) + max(right[0], right[1]), node.Val + left[0] + right[0]}
//	}
//	res := dfs(root)
//	return max(res[0], res[1])
//}

//股票系列1 只能买卖一次
//func maxProfit(prices []int) int {
//	//[0]:持有股票的最大现金
//	//[1]:不持有股票的最大现金
//	dp := make([][2]int, len(prices))
//	dp[0][0] = -prices[0]
//	for i := 1; i < len(prices); i++ {
//		//股票只能买卖一次 从不持有到持有只能当天买入且是第一次买入
//		dp[i][0] = max(dp[i-1][0], -prices[i])
//		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
//	}
//	return max(dp[len(prices)-1][0], dp[len(prices)-1][1])
//}

// 股票系列2 多次买卖
//	func maxProfit(prices []int) int {
//		//[0]:持有股票
//		//[1]:不持有股票
//		dp := make([][2]int, len(prices))
//		dp[0][0] = -prices[0]
//		for i := 1; i < len(prices); i++ {
//			//在i天持有股票：i-1天已经持有、i天买入
//			dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
//			//i天不持有股票：i-1天不持有、i-1天持有i天卖出
//			dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
//		}
//		return max(dp[len(prices)-1][0], dp[len(prices)-1][1])
//	}
//

// 股票系列3 最多完成两次交易
//func maxProfit(prices []int) int {
//	//dp0 第一次持有~
//	//dp1 第一次卖出~
//	//dp2 第二次持有~
//	//dp3 第二次卖出~
//	dp := make([][4]int, len(prices))
//	dp[0][0] = -prices[0]
//	dp[0][2] = -prices[0]
//	for i := 1; i < len(prices); i++ {
//		dp[i][0] = max(dp[i-1][0], -prices[i])
//		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
//		dp[i][2] = max(dp[i-1][2], dp[i-1][1]-prices[i])
//		dp[i][3] = max(dp[i-1][3], dp[i-1][2]+prices[i])
//	}
//	return dp[len(prices)-1][3]
//}

// 股票系列4: 参照股票3完成 最多完成k次交易
//func maxProfit(k int, prices []int) int {
//	dp := make([][]int, 0, len(prices))
//	for i := 0; i < len(prices); i++ {
//		dp = append(dp, make([]int, 2*k))
//	}
//	for i := 0; i < 2*k; i += 2 {
//		dp[0][i] = -prices[0]
//	}
//	for i := 1; i < len(prices); i++ {
//		for j := 0; j < 2*k; j++ {
//			if j == 0 {
//				dp[i][j] = max(dp[i-1][j], -prices[i])
//			} else {
//				if j%2 == 0 {
//					dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]-prices[i])
//				} else {
//					dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]+prices[i])
//				}
//			}
//		}
//	}
//	return dp[len(prices)-1][2*k-1]
//}

// 股票系列5 有冷冻期
//func maxProfit(prices []int) int {
//	/**
//	dp0持有
//	dp1保持卖出状态
//	dp2卖出
//	dp3冷冻期
//	*/
//	dp := make([][4]int, len(prices))
//	dp[0][0] = -prices[0]
//	for i := 1; i < len(prices); i++ {
//		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i], dp[i-1][3]-prices[i])
//		dp[i][1] = max(dp[i-1][1], dp[i-1][3])
//		dp[i][2] = dp[i-1][0] + prices[i]
//		dp[i][3] = dp[i-1][2]
//	}
//	return max(dp[len(prices)-1][1], dp[len(prices)-1][2], dp[len(prices)-1][3])
//}

/*
01背包 先遍历物品再反向遍历背包
完全背包的组合问题 先遍历物品再正向遍历背包
完全背包的排列问题 先正向遍历背包再遍历物品
*/

// 股票系列6 含手续费
func maxProfit(prices []int, fee int) int {
	/**
	dp0 持有股票
	dp1 保持卖出状态
	dp2 卖出股票
	*/
	dp := make([][3]int, len(prices))
	dp[0][0] = -prices[0] - fee
	for i := 1; i < len(prices); i++ {
		//昨天卖出今天再买/昨天保持卖出今天买
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i]-fee, dp[i-1][2]-prices[i]-fee)
		//昨天刚卖/保持卖出
		dp[i][1] = max(dp[i-1][1], dp[i-1][2])
		dp[i][2] = dp[i-1][0] + prices[i]
	}
	return max(dp[len(prices)-1][1], dp[len(prices)-1][2])
}

// O(n^2)
func lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	for i := 1; i < len(nums); i++ {
		for j := i - 1; j >= 0; j-- {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
	}
	return slices.Max(dp)
}

func findLengthOfLCIS(nums []int) int {
	maxLen, length := 1, 1
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			length++
			maxLen = max(maxLen, length)
		} else {
			length = 1
		}
	}
	return maxLen
}

// 二维dp 两个输入地位等同 无法压缩
func findLength(nums1 []int, nums2 []int) int {
	dp := make([][]int, 0, len(nums1))
	ans := 0
	for i := 0; i < len(nums1); i++ {
		dp = append(dp, make([]int, len(nums2)))
	}
	for i := 0; i < len(nums1); i++ {
		if nums1[i] == nums2[0] {
			dp[i][0] = 1
			ans = 1
		}
	}
	for i := 0; i < len(nums2); i++ {
		if nums2[i] == nums1[0] {
			dp[0][i] = 1
			ans = 1
		}
	}
	for i := 1; i < len(nums1); i++ {
		for j := 1; j < len(nums2); j++ {
			if nums1[i] == nums2[j] {
				dp[i][j] = dp[i-1][j-1] + 1
				ans = max(ans, dp[i][j])
			}
		}
	}
	return ans
}
func longestCommonSubsequence(text1 string, text2 string) int {
	dp := make([][]int, 0, len(text1)+1)
	for i := 0; i < len(text1)+1; i++ {
		dp = append(dp, make([]int, len(text2)+1))
	}
	ans := 0
	for i := 1; i <= len(text1); i++ {
		for j := 1; j <= len(text2); j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
			ans = max(ans, dp[i][j])
		}
	}
	return ans
}
func maxUncrossedLines(nums1 []int, nums2 []int) int {
	dp := make([][]int, 0, len(nums1)+1)
	for i := 0; i < len(nums1)+1; i++ {
		dp = append(dp, make([]int, len(nums2)+1))
	}
	for i := 0; i < len(nums1); i++ {
		for j := 0; j < len(nums2); j++ {
			if nums1[i] == nums2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}

func isSubsequence(s string, t string) bool {
	dp := make([][]int, 0, len(s)+1)
	for i := 0; i < len(s)+1; i++ {
		dp = append(dp, make([]int, len(t)+1))
	}
	ans := 0
	for i := 1; i <= len(s); i++ {
		for j := 1; j <= len(t); j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
			ans = max(ans, dp[i][j])
		}
	}
	return ans == len(s)
}
func numDistinct(s string, t string) int {
	dp := make([][]int, 0, len(t)+1)
	for i := 0; i < len(t)+1; i++ {
		dp = append(dp, make([]int, len(s)+1))
	}
	for i := 0; i < len(s); i++ {
		dp[1][i+1] = dp[1][i]
		if s[i] == t[0] {
			dp[1][i+1] = dp[1][i] + 1
		}
	}
	for i := 1; i < len(t); i++ {
		for j := 0; j < len(s); j++ {
			if t[i] == s[j] {
				dp[i+1][j+1] = dp[i][j] + dp[i+1][j]
			} else {
				dp[i+1][j+1] = dp[i+1][j]
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}
func minimumRemoval(beans []int) int64 {
	slices.Sort(beans)
	suffix := make([]int, len(beans))
	for i := len(suffix) - 2; i >= 0; i-- {
		suffix[i] = suffix[i+1] + beans[i+1]
	}
	var ans int64 = math.MaxInt
	sum := 0
	for i := 0; i < len(beans); i++ {
		ans = min(ans, int64(sum+suffix[i]-beans[i]*(len(beans)-i-1)))
		sum += beans[i]
	}
	return ans
}

// 其实还是最长公共子序列
//
//	func minDistance(word1 string, word2 string) int {
//		dp := make([][]int, 0, len(word1)+1)
//		for i := 0; i < len(word1)+1; i++ {
//			dp = append(dp, make([]int, len(word2)+1))
//		}
//		for i := 0; i <= len(word1); i++ {
//			dp[i][0] = i
//		}
//		for i := 0; i <= len(word2); i++ {
//			dp[0][i] = i
//		}
//		for i := 0; i < len(word1); i++ {
//			for j := 0; j < len(word2); j++ {
//				if word1[i] == word2[j] {
//					dp[i+1][j+1] = dp[i][j]
//				} else {
//					dp[i+1][j+1] = min(dp[i+1][j]+1, dp[i][j+1]+1, dp[i][j]+2)
//				}
//			}
//		}
//		return dp[len(dp)-1][len(dp[0])-1]
//	}

func minDistance(word1 string, word2 string) int {
	dp := make([][]int, 0, len(word1)+1)
	for i := 0; i < len(word1)+1; i++ {
		dp = append(dp, make([]int, len(word2)+1))
	}
	for i := 0; i <= len(word1); i++ {
		dp[i][0] = i
	}
	for i := 0; i <= len(word2); i++ {
		dp[0][i] = i
	}
	for i := 0; i < len(word1); i++ {
		for j := 0; j < len(word2); j++ {
			if word1[i] == word2[j] {
				dp[i+1][j+1] = dp[i][j]
			} else {
				dp[i+1][j+1] = min(dp[i+1][j]+1, dp[i][j+1]+1, dp[i][j]+1)
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}
func isPalindrome(s string) bool {
	left := 0
	right := len(s) - 1
	for left <= right {
		if s[left:left+1] != s[right:right+1] {
			return false
		}
		left++
		right--
	}
	return true
}

//func countSubstrings(s string) int {
//	dp := make([]int, len(s))
//	dp[0] = 1
//	for i := 1; i < len(s); i++ {
//		dp[i] = dp[i-1] + 1
//		for j := 0; j < i; j++ {
//			if isPalindrome(s[j : i+1]) {
//				dp[i]++
//			}
//		}
//	}
//	return dp[len(dp)-1]
//}

func countSubstrings(s string) int {
	count := 0
	dp := make([][]bool, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
	}
	for j := 0; j < len(s); j++ {
		for i := 0; i <= j; i++ {
			if i == j {
				dp[i][j] = true
				count++
			} else if j-i == 1 && s[i] == s[j] {
				dp[i][j] = true
				count++
			} else if j-i > 1 && s[i] == s[j] && dp[i+1][j-1] {
				dp[i][j] = true
				count++
			}
		}
	}
	return count
}

func longestPalindromeSubseq(s string) int {
	dp := make([][]int, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(s))
	}
	for i := len(dp) - 1; i >= 0; i-- {
		for j := i; j < len(dp); j++ {
			if j == i {
				dp[i][j] = 1
			} else if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = max(dp[i+1][j-1], dp[i+1][j], dp[i][j-1])
			}
		}
	}
	return dp[0][len(dp)-1]
}

func longestPalindrome(s string) string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
	}
	palindrome := ""
	for i := len(dp) - 1; i >= 0; i-- {
		for j := i; j < len(dp); j++ {
			if s[i] == s[j] && (j-i <= 1 || dp[i+1][j-1]) {
				dp[i][j] = true
				if j-i+1 > len(palindrome) {
					palindrome = s[i : j+1]
				}
			}
		}
	}
	return palindrome
}

func canEat(piles []int, speed, h int) bool {
	cnt := 0
	for _, pile := range piles {
		cnt += (pile + speed - 1) / speed
	}
	return cnt <= h
}

func minEatingSpeed(piles []int, h int) int {
	minSpeed := 1
	maxSpeed := slices.Max(piles)
	return sort.Search(maxSpeed, func(i int) bool {
		return canEat(piles, i+minSpeed, h)
	}) + minSpeed
}

func canSplit(nums []int, k, maxSum int) bool {
	sum := 0
	index := 0
	split := 1
	for sum <= maxSum && index < len(nums) {
		if sum+nums[index] > maxSum {
			split++
			sum = 0
			continue
		}
		sum += nums[index]
		index++
	}
	return split <= k
}

// 画匠问题
// 转换思路：如何划分为k组使其最大和最小=>给定一个最大和能否划分为k组(划分成多少份就可以 splits<k)
// 最大值最小、最小值最大一定是二分(求最小值、最大值也可能是)
// 确定答案可能存在的范围 在答案的范围上不断二分 寻找到最小(最大)的满足条件的结果
// 从求解答案变成判断一个答案能否满足条件
func splitArray(nums []int, k int) int {
	var check func(sum int) bool
	check = func(sum int) bool {
		cnt := 1
		curSum := 0
		index := 0
		for index < len(nums) {
			if curSum+nums[index] <= sum {
				curSum += nums[index]
				index++
			} else {
				curSum = 0
				cnt++
				if cnt > k {
					return false
				}
			}
		}
		return cnt <= k
	}
	maxSum := 0
	for _, num := range nums {
		maxSum += num
	}
	return sort.Search(maxSum, check)
}

// 堆超时 双重循环遍历数组的过程O(n^2)超时
//
//	func smallestDistancePair(nums []int, k int) int {
//		hp := &IntHeap{}
//		for i := 0; i < len(nums); i++ {
//			for j := i + 1; j < len(nums); j++ {
//				val := Abs(nums[i] - nums[j])
//				heap.Push(hp, val)
//				if hp.Len() > k {
//					heap.Pop(hp)
//				}
//			}
//		}
//		return heap.Pop(hp).(int)
//	}

func smallestDistancePair(nums []int, k int) int {
	var fn func(dist int) bool
	fn = func(dist int) bool {
		cnt := 0
		for i, j := 0, 0; i < len(nums); i++ {
			for j < len(nums) && nums[j]-nums[i] <= dist {
				j++
			}
			cnt += j - i - 1
		}
		return cnt >= k
	}
	slices.Sort(nums)
	return sort.Search(nums[len(nums)-1]-nums[0], fn)
}
func repairCars(ranks []int, cars int) int64 {
	var canRepair func(time int) bool
	canRepair = func(time int) bool {
		cnt := 0
		for _, rank := range ranks {
			cnt += int(math.Sqrt(float64(time / rank)))
			if cnt >= cars {
				return true
			}
		}
		return cnt >= cars
	}
	maxTime := slices.Min(ranks) * cars * cars
	return int64(sort.Search(maxTime, canRepair))
}
