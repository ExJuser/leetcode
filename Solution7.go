package main

func countGood(nums []int, k int) (ans int64) {
	var left, pairCount int
	cnt := make(map[int]int)
	for right := 0; right < len(nums); right++ {
		pairCount += cnt[nums[right]]
		cnt[nums[right]]++
		for ; pairCount >= k; left++ {
			cnt[nums[left]]--
			pairCount -= cnt[nums[left]]
			ans += int64(len(nums) - right)
		}
	}
	return
}

// 窗口内的最大最小值的差不大于2
// 单调队列维护最大最小值
func continuousSubarrays(nums []int) (ans int64) {
	maxQueue := make([]int, 0, len(nums))
	minQueue := make([]int, 0, len(nums))
	var left int
	for right := 0; right < len(nums); right++ {
		for len(maxQueue) > 0 && maxQueue[len(maxQueue)-1] < nums[right] {
			maxQueue = maxQueue[:len(maxQueue)-1]
		}
		for len(minQueue) > 0 && minQueue[len(minQueue)-1] > nums[right] {
			minQueue = minQueue[:len(minQueue)-1]
		}
		minQueue = append(minQueue, nums[right])
		maxQueue = append(maxQueue, nums[right])
		for ; len(maxQueue) > 0 && len(minQueue) > 0 && maxQueue[0]-minQueue[0] > 2; left++ {
			if maxQueue[0] == nums[left] {
				maxQueue = maxQueue[1:]
			}
			if minQueue[0] == nums[left] {
				minQueue = minQueue[1:]
			}
		}
		ans += int64(right - left + 1)
	}
	return
}
