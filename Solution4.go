package main

// 最长重复子数组：连续
func findLength(nums1 []int, nums2 []int) (ans int) {
	dp := make([][]int, len(nums1))
	for i := 0; i < len(nums1); i++ {
		dp[i] = make([]int, len(nums2))
	}
	for i := 0; i < len(nums1); i++ {
		if nums1[i] == nums2[0] {
			dp[i][0] = 1
			ans = 1
		}
	}
	for j := 0; j < len(nums2); j++ {
		if nums2[j] == nums1[0] {
			dp[0][j] = 1
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
	return
}

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

func longestCommonSubsequence(text1 string, text2 string) (ans int) {
	dp := make([][]int, len(text1))
	for i := 0; i < len(text1); i++ {
		dp[i] = make([]int, len(text2))
	}
	for i := 0; i < len(text1); i++ {
		if text1[i] == text2[0] {
			for j := i; j < len(text1); j++ {
				dp[j][0] = 1
				ans = 1
			}
			break
		}
	}
	for j := 0; j < len(text2); j++ {
		if text2[j] == text1[0] {
			for i := j; i < len(text2); i++ {
				dp[0][i] = 1
				ans = 1
			}
		}
	}
	for i := 1; i < len(text1); i++ {
		for j := 1; j < len(text2); j++ {
			if text1[i] == text2[j] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
			ans = max(ans, dp[i][j])
		}
	}
	return ans
}
