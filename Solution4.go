package main

import (
	"container/heap"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// 最长重复子数组：连续
//func findLength(nums1 []int, nums2 []int) (ans int) {
//	dp := make([][]int, len(nums1))
//	for i := 0; i < len(nums1); i++ {
//		dp[i] = make([]int, len(nums2))
//	}
//	for i := 0; i < len(nums1); i++ {
//		if nums1[i] == nums2[0] {
//			dp[i][0] = 1
//			ans = 1
//		}
//	}
//	for j := 0; j < len(nums2); j++ {
//		if nums2[j] == nums1[0] {
//			dp[0][j] = 1
//			ans = 1
//		}
//	}
//	for i := 1; i < len(nums1); i++ {
//		for j := 1; j < len(nums2); j++ {
//			if nums1[i] == nums2[j] {
//				dp[i][j] = dp[i-1][j-1] + 1
//				ans = max(ans, dp[i][j])
//			}
//		}
//	}
//	return
//}

// 最长公共子序列：可以不连续
//func longestCommonSubsequence(text1 string, text2 string) (ans int) {
//	dp := make([][]int, len(text1)+1)
//	for i := 0; i <= len(text1); i++ {
//		dp[i] = make([]int, len(text2)+1)
//	}
//	for i := 0; i < len(text1); i++ {
//		for j := 0; j < len(text2); j++ {
//			if text1[i] == text2[j] {
//				dp[i+1][j+1] = dp[i][j] + 1
//			} else {
//				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
//			}
//			ans = max(ans, dp[i+1][j+1])
//		}
//	}
//	return
//}

//func longestCommonSubsequence(text1 string, text2 string) (ans int) {
//	dp := make([][]int, len(text1))
//	for i := 0; i < len(text1); i++ {
//		dp[i] = make([]int, len(text2))
//	}
//	for i := 0; i < len(text1); i++ {
//		if text1[i] == text2[0] {
//			for j := i; j < len(text1); j++ {
//				dp[j][0] = 1
//				ans = 1
//			}
//			break
//		}
//	}
//	for j := 0; j < len(text2); j++ {
//		if text2[j] == text1[0] {
//			for i := j; i < len(text2); i++ {
//				dp[0][i] = 1
//				ans = 1
//			}
//		}
//	}
//	for i := 1; i < len(text1); i++ {
//		for j := 1; j < len(text2); j++ {
//			if text1[i] == text2[j] {
//				dp[i][j] = dp[i-1][j-1] + 1
//			} else {
//				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
//			}
//			ans = max(ans, dp[i][j])
//		}
//	}
//	return ans
//}

// 不相交的线：最长公共子序列
func maxUncrossedLines(nums1 []int, nums2 []int) (ans int) {
	dp := make([][]int, len(nums1)+1)
	for i := 0; i <= len(nums1); i++ {
		dp[i] = make([]int, len(nums2)+1)
	}
	for i := 0; i < len(nums1); i++ {
		for j := 0; j < len(nums2); j++ {
			if nums1[i] == nums2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
			}
			ans = max(ans, dp[i+1][j+1])
		}
	}
	return
}

// 最大子数组和：连续
func maxSubArray1(nums []int) int {
	dp := make([]int, len(nums))
	copy(dp, nums)
	ans := dp[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(dp[i], nums[i]+dp[i-1])
		ans = max(ans, dp[i])
	}
	return ans
}

// 判断子序列
// 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
//func isSubsequence(s string, t string) bool {
//	if len(s) == 0 {
//		return true
//	}
//	if len(t) == 0 {
//		return false
//	}
//	dp := make([][]bool, len(s))
//	for i := 0; i < len(dp); i++ {
//		dp[i] = make([]bool, len(t))
//	}
//	for j := 0; j < len(t); j++ {
//		if s[0] == t[j] {
//			for i := j; i < len(t); i++ {
//				dp[0][i] = true
//			}
//			break
//		}
//	}
//	//如果相等 dp[i][j]=dp[i-1][j-1]
//	//如果不相等 dp[i][j]=dp[i][j-1]
//	for i := 1; i < len(s); i++ {
//		for j := 1; j < len(t); j++ {
//			if s[i] == t[j] {
//				dp[i][j] = dp[i-1][j-1]
//			} else {
//				dp[i][j] = dp[i][j-1]
//			}
//		}
//	}
//	return dp[len(s)-1][len(t)-1]
//}

// 不同的子序列
// 返回在s的子序列中t出现的个数
//func numDistinct(s string, t string) int {
//	dp := make([][]int, len(s)+1)
//	for i := 0; i < len(dp); i++ {
//		dp[i] = make([]int, len(t)+1)
//	}
//	for i := 0; i < len(dp); i++ {
//		dp[i][0] = 1
//	}
//	for i := 1; i < len(dp); i++ {
//		for j := 1; j < len(dp[i]); j++ {
//			if s[i-1] == t[j-1] {
//				dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
//			} else {
//				dp[i][j] = dp[i-1][j]
//			}
//		}
//	}
//	return dp[len(s)][len(t)]
//}

func numDistinct(s string, t string) int {
	dp := make([][]int, len(s)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(t)+1)
	}
	//初始化 dp[i][0] dp[0][j]
	for i := 0; i < len(s); i++ {
		dp[i][0] = 1
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}
	return dp[len(s)][len(t)]
}

// 判断子序列：判断s是否为t的子序列
func isSubsequence(s string, t string) bool {
	dp := make([][]int, len(s)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(t)+1)
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}
	return dp[len(s)][len(t)] == len(s)
}

// 最长公共子序列
func longestCommonSubsequence(text1 string, text2 string) int {
	dp := make([][]int, len(text1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(text2)+1)
	}
	//不需要初始化 因为一个字符串和一个空串的最长公共子序列长度一定为0
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[len(text1)][len(text2)]
}

// 最长重复子数组
func findLength(nums1 []int, nums2 []int) (ans int) {
	dp := make([][]int, len(nums1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(nums2)+1)
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if nums1[i-1] == nums2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			}
			ans = max(ans, dp[i][j])
		}
	}
	return
}

// 两个字符串的删除操作
// 使得 word1 和 word2 相同所需的最小步数
//
//	func minDistance(word1 string, word2 string) int {
//		//dp[i][j] 代表使得以i-1结尾的word1和j-1结尾的word2相同的最小步数
//		dp := make([][]int, len(word1)+1)
//		for i := 0; i < len(dp); i++ {
//			dp[i] = make([]int, len(word2)+1)
//		}
//		//初始化 dp[i][0]
//		//使得一个空串和一个字符串相等的最小步数即字符串长度
//		for i := 0; i < len(dp); i++ {
//			dp[i][0] = i
//		}
//		for j := 0; j < len(dp[0]); j++ {
//			dp[0][j] = j
//		}
//		for i := 1; i < len(dp); i++ {
//			for j := 1; j < len(dp[i]); j++ {
//				if word1[i-1] == word2[j-1] {
//					//两个字符相同 不需要操作
//					dp[i][j] = dp[i-1][j-1]
//				} else {
//					dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1
//				}
//			}
//		}
//		return dp[len(word1)][len(word2)]
//	}

// 编辑距离
func minDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(word2)+1)
	}
	//初始化 dp[i][0]
	//使得一个空串和一个字符串相等的最小步数即字符串长度
	for i := 0; i < len(dp); i++ {
		dp[i][0] = i
	}
	for j := 0; j < len(dp[0]); j++ {
		dp[0][j] = j
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
			}
		}
	}
	return dp[len(word1)][len(word2)]
}

func countSubstrings(s string) int {
	count := 0
	dp := make([][]bool, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
	}
	for i := len(s) - 1; i >= 0; i-- {
		for j := i; j < len(s); j++ {
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

// 最长回文子序列
func longestPalindromeSubseq(s string) int {
	//dp[i][j] 以i开头j结尾的字符串中最长的回文子序列长度
	dp := make([][]int, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(s))
	}
	for i := len(s) - 1; i >= 0; i-- {
		for j := i; j < len(s); j++ {
			if i == j {
				dp[i][j] = 1
			} else if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = max(dp[i+1][j], dp[i][j-1])
			}
		}
	}
	return dp[0][len(s)-1]
}

// 旋转图像：先按照对角线交换 然后每一行逆序
func rotate(matrix [][]int) {
	for i := 0; i < len(matrix); i++ {
		for j := i; j < len(matrix[i]); j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
	for i := 0; i < len(matrix); i++ {
		slices.Reverse(matrix[i])
	}
}

// 非原地算法解决
func gameOfLife(board [][]int) {
	newBoard := make([][]int, len(board))
	for i := 0; i < len(newBoard); i++ {
		newBoard[i] = make([]int, len(board[i]))
	}
	copy(newBoard, board)
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			count := 0
			for m := i - 1; m >= 0 && m <= i+1 && m < len(board); m++ {
				for n := j - 1; n >= 0 && n <= j+1 && n < len(board[i]); n++ {
					if m != i && n != j && board[m][n] == 1 {
						count++
					}
				}
			}
			//活细胞少于两个  活细胞死亡、死细胞仍死亡
			//活细胞2个 	   活细胞存活、死细胞仍死亡
			//活细胞3个      活细胞存活、死细胞复活
			//活细胞超过三个  活细胞死亡 死细胞复活
			//即活细胞死亡的情况为：活细胞少于两个、活细胞超过三个
			//死细胞复活的情况为：大于等于三个
			if board[i][j] == 1 && (count < 2 || count > 3) {
				newBoard[i][j] = 0
			}
			if board[i][j] == 0 && count >= 3 {
				newBoard[i][j] = 1
			}
		}
	}
}
func isValid1(s string) bool {
	stack := make([]byte, 0)
	for _, ch := range s {
		if strings.Contains("([{", string(ch)) {
			stack = append(stack, byte(ch))
		} else {
			if len(stack) == 0 || ch == ')' && stack[len(stack)-1] != '(' ||
				(ch == ']' && stack[len(stack)-1] != '[') ||
				(ch == '}' && stack[len(stack)-1] != '{') {
				return false
			} else {
				stack = stack[:len(stack)-1]
			}
		}
	}
	return len(stack) == 0
}

// 二叉树的中序遍历
func inorderTraversal1(root *TreeNode) (ans []int) {
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

func removeKdigits(num string, k int) string {
	stack := make([]byte, 0)
	for i := 0; i < len(num); i++ {
		for len(stack) > 0 && stack[len(stack)-1]-'0' > num[i]-'0' {
			if k == 0 {
				break
			}
			k--
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, num[i])
	}
	for k > 0 {
		stack = stack[:len(stack)-1]
		k--
	}
	ans := strings.TrimLeft(string(stack), "0")
	if ans == "" {
		return "0"
	}
	return ans
}

// 移除重复字母且使得字典序最小
func removeDuplicateLetters(s string) string {
	count := make(map[byte]int)
	for _, char := range s {
		count[byte(char)]++
	}
	stack := make([]byte, 0)
	for _, char := range s {
		if !slices.Contains(stack, byte(char)) {
			for len(stack) > 0 && stack[len(stack)-1] >= byte(char) && count[stack[len(stack)-1]] >= 1 {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, byte(char))
		}
		count[byte(char)]--
	}
	return string(stack)
}

func vowelStrings(words []string, queries [][]int) (ans []int) {
	var isVowelString func(word string) bool
	isVowelString = func(word string) bool {
		start, end := string(word[0]), string(word[len(word)-1])
		if strings.Contains("aeiou", start) && strings.Contains("aeiou", end) {
			return true
		}
		return false
	}
	prefix := make([]int, len(words)+1)
	for i := 0; i < len(words); i++ {
		prefix[i+1] = prefix[i]
		if isVowelString(words[i]) {
			prefix[i+1]++
		}
	}
	for _, query := range queries {
		ans = append(ans, prefix[query[1]+1]-prefix[query[0]])
	}
	return
}

func answerQueries(nums []int, queries []int) []int {
	slices.Sort(nums)
	prefix := make([]int, len(nums)+1)
	ans := make([]int, len(queries))
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] + nums[i]
	}
	for i, query := range queries {
		ans[i] = sort.Search(len(prefix), func(i int) bool {
			return prefix[i] > query
		}) - 1
	}
	return ans
}

func productQueries(n int, queries [][]int) (ans []int) {
	var getPowers func(n int) []int
	getPowers = func(n int) []int {
		powers := make([]int, 0)
		base := 1
		for ; n > 0; n >>= 1 {
			if n&1 == 1 {
				powers = append(powers, base)
			}
			base *= 2
		}
		return powers
	}
	powers := getPowers(n)
	for _, query := range queries {
		res := 1
		for i := query[0]; i <= query[1]; i++ {
			res = res * powers[i] % (1e9 + 7)
		}
		ans = append(ans, res)
	}
	return
}

func numSubarraysWithSum(nums []int, goal int) (ans int) {
	prefix := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] + nums[i]
	}
	cnt := make(map[int]int)
	for i := 0; i < len(prefix); i++ {
		cnt[prefix[i]]++
	}
	for k, _ := range cnt {
		if k >= goal {
			if k == k-goal {
				ans += cnt[k] * (cnt[k] - 1) / 2
			} else {
				ans += cnt[k] * cnt[k-goal]
			}
		}
	}
	return
}

// 和为k的子数组数目
func subarraySum(nums []int, k int) (ans int) {
	prefix := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] + nums[i]
	}
	cnt := make(map[int]int)
	for i := 0; i < len(prefix); i++ {
		ans += cnt[prefix[i]-k]
		cnt[prefix[i]]++
	}
	return ans
}

func numOfSubarrays(arr []int) (ans int) {
	prefix := make([]int, len(arr)+1)
	for i := 0; i < len(arr); i++ {
		prefix[i+1] = prefix[i] + arr[i]
	}
	var OddNum, evenNum int
	for i := 0; i < len(prefix); i++ {
		if prefix[i]%2 == 0 {
			ans = (ans + OddNum) % (1e9 + 7)
			evenNum++
		} else {
			ans = (ans + evenNum) % (1e9 + 7)
			OddNum++
		}
	}
	return
}

func subarraysDivByK(nums []int, k int) (ans int) {
	prefix := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		//不管用什么方法 目的都是让“前缀和”数组均为正数
		prefix[i+1] = (prefix[i]%k + nums[i]%k + k) % k
		//if prefix[i+1] < 0 {
		//	prefix[i+1] += k
		//}
	}
	//for i := 0; i < len(prefix); i++ {
	//	//可能不够 这一步的目的就是让其变成正的：同余定理
	//	//因此直接在上面一步处理
	//	prefix[i] = (prefix[i] + k) % k
	//}
	cnt := make(map[int]int)
	for i := 0; i < len(prefix); i++ {
		//没有想到同余定理时需要考虑的情况太多
		//ans += cnt[prefix[i]] + cnt[prefix[i]-k] + cnt[prefix[i]+k]
		ans += cnt[prefix[i]]
		cnt[prefix[i]]++
	}
	return
}

type FindElements struct {
	cnt map[int]struct{}
}

//func Constructor(root *TreeNode) FindElements {
//	var dfs func(node *TreeNode, val int)
//	cnt := make(map[int]struct{})
//	dfs = func(node *TreeNode, val int) {
//		if node == nil {
//			return
//		}
//		node.Val = val
//		cnt[val] = struct{}{}
//		dfs(node.Left, 2*node.Val+1)
//		dfs(node.Right, 2*node.Val+2)
//	}
//	dfs(root, 0)
//	return FindElements{cnt: cnt}
//}

func (this *FindElements) Find(target int) bool {
	if _, ok := this.cnt[target]; ok {
		return true
	}
	return false
}

// 从前序与中序遍历序列构造二叉树
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) > 0 {
		rootVal := preorder[0]
		leftLength := slices.Index(inorder, rootVal)
		left := buildTree(preorder[1:1+leftLength], inorder[:leftLength+1])
		right := buildTree(preorder[1+leftLength:], inorder[leftLength+1:])
		return &TreeNode{Val: rootVal, Left: left, Right: right}
	}
	return nil
}

// 滑动窗口：区间子数组个数
func numSubarrayBoundedMax(nums []int, left int, right int) (ans int) {
	//对于每一位 第一个大于等于left和第一个大于right的之间的元素个数就是答案
	l, r := -1, -1
	for i := 0; i < len(nums); i++ {
		if nums[i] >= left {
			l = i
		}
		if nums[i] > right {
			r = i
		}
		ans += l - r
	}
	return
}

// 下降路径最小和
func minFallingPathSum(matrix [][]int) int {
	dp := make([][]int, len(matrix))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(matrix[i]))
	}
	for i := 0; i < len(dp[0]); i++ {
		dp[0][i] = matrix[0][i]
	}
	for i := 1; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			dp[i][j] = dp[i-1][j] + matrix[i][j]
			if j-1 >= 0 {
				dp[i][j] = min(dp[i][j], dp[i-1][j-1]+matrix[i][j])
			}
			if j+1 < len(matrix[i]) {
				dp[i][j] = min(dp[i][j], dp[i-1][j+1]+matrix[i][j])
			}
		}
	}
	return slices.Min(dp[len(matrix)-1])
}

// 常规dp+前缀和：复杂度比较高
//func maxSubarraySumCircular(nums []int) int {
//	if len(nums) == 1 {
//		return nums[0]
//	}
//	dp := make([]int, len(nums))
//	dp[0] = nums[0]
//	prefix := make([]int, len(nums)+1)
//	for i := 1; i < len(nums); i++ {
//		dp[i] = max(nums[i], dp[i-1]+nums[i])
//		prefix[i] = prefix[i-1] + nums[i-1]
//	}
//	prefix[len(nums)] = prefix[len(nums)-1] + nums[len(nums)-1]
//	dpMax := slices.Max(dp)
//	prefixMax := 0
//	for i := 1; i < len(nums); i++ {
//		dpMax = max(dpMax, prefix[len(nums)]-prefix[i]+max(prefixMax, prefix[i]))
//	}
//	return dpMax
//}

// 存在环形的最大和=数组和-最小子数组和
func maxSubarraySumCircular(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	dpMin := make([]int, len(nums))
	dpMax := make([]int, len(nums))
	dpMin[0], dpMax[0] = nums[0], nums[0]
	arrSum := nums[0]
	for i := 1; i < len(nums); i++ {
		dpMax[i] = max(dpMax[i-1]+nums[i], nums[i])
		dpMin[i] = min(dpMin[i-1]+nums[i], nums[i])
		arrSum += nums[i]
	}
	maxS, minS := slices.Max(dpMax), slices.Min(dpMin)
	if arrSum == minS {
		return maxS
	}
	return max(maxS, arrSum-minS)
}

func maxSubArray2(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	ans := nums[0]
	for i := 1; i < len(nums); i++ {
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		ans = max(ans, dp[i])
	}
	return ans
}

// 再次应用到了同余原理
func checkSubarraySum(nums []int, k int) bool {
	prefix := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = (prefix[i] + nums[i]) % k
	}
	mp := make(map[int]int)
	mp[0] = -1
	for i := 0; i < len(nums); i++ {
		if index, ok := mp[prefix[i+1]]; ok && i-index >= 2 {
			return true
		} else if !ok {
			mp[prefix[i+1]] = i
		}
	}
	return false
}

// prefix其实也可以优化掉
func findMaxLength(nums []int) int {
	prefix := make([]int, len(nums)+1)
	mp := map[int]int{0: -1}
	ans := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			prefix[i+1] = prefix[i] + 1
		} else {
			prefix[i+1] = prefix[i] - 1
		}
		if index, ok := mp[prefix[i+1]]; ok {
			ans = max(ans, i-index)
		} else {
			mp[prefix[i+1]] = i
		}
	}
	return ans
}

// 最具竞争力的子序列
func mostCompetitive(nums []int, k int) []int {
	stack := make([]int, 0, len(nums))
	ans := make([]int, k)
	for i, num := range nums {
		for len(stack) > 0 && stack[len(stack)-1] > num && len(nums)-i+len(stack) > k {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, num)
		if len(stack) == k {
			ans = append([]int{}, stack...)
		}
	}
	return ans
}

// 定长滑动窗口
func maxVowels(s string, k int) (ans int) {
	length := 0
	for i := 0; i < k; i++ {
		if strings.Contains("aeiou", string(s[i])) {
			length++
		}
	}
	ans = max(ans, length)
	for i := 1; i < len(s)-k+1; i++ {
		if strings.Contains("aeiou", string(s[i-1])) {
			length--
		}
		if strings.Contains("aeiou", string(s[i+k-1])) {
			length++
		}
		ans = max(ans, length)
	}
	return
}

func findMaxAverage(nums []int, k int) float64 {
	var sum, maxSum int
	for i := 0; i < k; i++ {
		sum += nums[i]
	}
	maxSum = sum
	for i := 1; i < len(nums)-k+1; i++ {
		sum -= nums[i-1] - nums[i+k-1]
		maxSum = max(sum, maxSum)
	}
	return float64(maxSum) / float64(k)
}

func divisorSubstrings(num int, k int) int {
	numStr := strconv.Itoa(num)
	ans := 0
	for i := 0; i < len(numStr)-k+1; i++ {
		if val, _ := strconv.Atoi(numStr[i : i+k]); val != 0 && num%val == 0 {
			ans++
		}
	}
	return ans
}

func minimumDifference(nums []int, k int) int {
	ans := math.MaxInt
	slices.Sort(nums)
	for i := 0; i < len(nums)-k+1; i++ {
		ans = min(ans, nums[i+k-1]-nums[i])
	}
	return ans
}

func numOfSubarrays1(arr []int, k int, threshold int) int {
	var sum, ans int
	for i := 0; i < k; i++ {
		sum += arr[i]
	}
	if sum >= threshold*k {
		ans = 1
	}
	for i := 1; i < len(arr)-k+1; i++ {
		sum -= arr[i-1] - arr[i+k-1]
		if sum >= threshold*k {
			ans++
		}
	}
	return ans
}

func getAverages(nums []int, k int) []int {
	ans := make([]int, len(nums))
	for i := 0; i < len(ans); i++ {
		ans[i] = -1
	}
	if len(nums) < 2*k+1 {
		return ans
	}
	sum := 0
	for i := 0; i <= 2*k; i++ {
		sum += nums[i]
	}
	ans[k] = sum / (2*k + 1)
	for i := k + 1; i < len(nums)-k; i++ {
		sum -= nums[i-k-1] - nums[i+k]
		ans[i] = sum / (2*k + 1)
	}
	return ans
}

func findIndices(nums []int, indexDifference int, valueDifference int) []int {
	for i := 0; i < len(nums); i++ {
		for j := i; j < len(nums); j++ {
			if Abs(i-j) >= indexDifference && Abs(nums[i]-nums[j]) >= valueDifference {
				return []int{i, j}
			}
		}
	}
	return []int{-1, -1}
}

// 最后一块石头的重量
func lastStoneWeight(stones []int) int {
	hp := &IntHeap{}
	*hp = stones
	heap.Init(hp)
	for hp.Len() > 1 {
		y, x := heap.Pop(hp).(int), heap.Pop(hp).(int)
		if x != y {
			heap.Push(hp, y-x)
		}
	}
	if hp.Len() == 0 {
		return 0
	}
	return heap.Pop(hp).(int)
}
func pickGifts(gifts []int, k int) int64 {
	hp := &IntHeap{}
	*hp = gifts
	heap.Init(hp)
	for ; k > 0; k-- {
		gift := heap.Pop(hp).(int)
		heap.Push(hp, int(math.Sqrt(float64(gift))))
	}
	var ans int64
	for _, gift := range gifts {
		ans += int64(gift)
	}
	return ans
}

type SmallestInfiniteSet struct {
	hp       *IntHeap
	notExist map[int]struct{}
}

//func Constructor() SmallestInfiniteSet {
//	hp := &IntHeap{}
//	set := make([]int, 1000)
//	for i := 1; i <= 1000; i++ {
//		set[i-1] = i
//	}
//	*hp = set
//	heap.Init(hp)
//	return SmallestInfiniteSet{hp: hp, notExist: map[int]struct{}{}}
//}

func (this *SmallestInfiniteSet) PopSmallest() int {
	x := heap.Pop(this.hp).(int)
	this.notExist[x] = struct{}{}
	return x
}

func (this *SmallestInfiniteSet) AddBack(num int) {
	if _, ok := this.notExist[num]; ok {
		delete(this.notExist, num)
		heap.Push(this.hp, num)
	}
}

func maxKelements(nums []int, k int) (ans int64) {
	hp := &IntHeap{}
	*hp = nums
	heap.Init(hp)
	for ; k > 0; k-- {
		x := heap.Pop(hp).(int)
		ans += int64(x)
		heap.Push(hp, int(math.Ceil(float64(x)/3)))
	}
	return
}
