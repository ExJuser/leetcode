package main

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
func numDistinct(s string, t string) int {
	dp := make([][]int, len(s)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(t)+1)
	}
	for i := 0; i < len(dp); i++ {
		dp[i][0] = 1
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
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
