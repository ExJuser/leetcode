package main

import (
	"math"
)

func maximalSquare(matrix [][]byte) int {
	dp := make([][]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
	}
	ans := 0
	for i := 0; i < len(dp); i++ {
		for j := 0; j < len(dp[i]); j++ {
			if matrix[i][j] == '1' {
				dp[i][j] = 1
				ans = max(ans, dp[i][j])
			}
		}
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if dp[i][j] == 1 {
				x := min(int(math.Sqrt(float64(dp[i-1][j]))), int(math.Sqrt(float64(dp[i-1][j-1]))), int(math.Sqrt(float64(dp[i][j-1])))) + 1
				dp[i][j] = x * x
				ans = max(ans, dp[i][j])
			}
		}
	}
	return ans
}
