| 方法       | 适用场景                 | 思路核心           | 时间复杂度      | 空间复杂度           |
| -------- | -------------------- | -------------- | ---------- | --------------- |
| **二分查找** | 有序数组 / 单调性问题         | 每次折半查找目标或边界    | `O(log n)` | `O(1)`          |
| **双指针法** | 有序数组去重、原地修改、和问题、快慢链表 | 两个指针同向或对撞移动    | `O(n)`     | `O(1)`          |
| **滑动窗口** | 连续子数组/子串问题，最长/最短区间   | 动态维护窗口（左右指针移动） | `O(n)`     | `O(1)` ~ `O(k)` |

# 顺序查找
列表中数据项并没有按值排列顺序，而是随机放置在列表中的。
```python
def sequential(alist, item):
    pos = 0
    found = False
    
    while pos < len(alist) and not found:
        if alist[pos] == item:
            found = True
        else:
            pos += 1
    
            return found
```

# 二分查找
条件：
- 数组为有序数组
- 数组中无重复元素

区间的定义一般为两种：
- 左闭右闭即`[left, right]`
- 左闭右开即`[left, right)`

## 区间定义
### `[left, right]`
- 左闭右闭的情况，left和right可以相等，所以是`while (left<=right)`
- 不断取中时，因为会有`mid == target`的判断，所以`mid != target`的时，右边边界为`mid-1`，左边边界为`mid+1`

### `[left, right)`
- 左闭右开，无法取到右边边界 `while (left < right)`
- 不断取中，右边边界缩小为`mid`，左边边界缩小时为`mid+1`

## time complexity
$O(\log n)$
1. 查找：每一步都会把搜索区间缩小一半
2. 每一步缩小问题规模： n -> n/2 -> n/4 -> ... -> n/$2^k$，直到最后只剩下1个元素，即$n/2^k = 1$
3. $n / 2^k = 1$ -> $n = 2^k $  -> $k=\log_2 n$

k即二分查找最多需要的比较次数


## 相关题目
[704.二分查找](https://leetcode.cn/problems/binary-search/description/)<br>
[35.搜索插入位置](https://leetcode.cn/problems/search-insert-position/description/)


# 双指针法
用一个快指针向前遍历数组，一个慢指针，用于修改元素

两个指针`i`和`k`：
- `i` 遍历整个数组
- `k` 指向下一个要写入的元素位置（即不等于 val 的元素应该放的位置）

遍历数组：
- 如果nums[i] != val:
  - 将nums[i]放到nums[k]
  - k += 1

## complexity
- time complexity: $O(n)$ 遍历数组
- space complexity: $O(1)$ 无需开辟新的空间

### 对比暴力解法
两层循环，第一层循环遍历数组，第二层循环移动元素

#### complexity
- time complexity: $O(n^2)$ 遍历数组
- space complexity: $O(1)$ 无需开辟新的空间

## 相关题目
[27.移除元素](https://leetcode.cn/problems/remove-element/submissions/667457571/)<br>
[26.删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)<br>
[283.移动零](https://leetcode.cn/problems/move-zeroes/)<br>

[977.有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/description/)<br>
这道题有多种解法：[解析](https://programmercarl.com/0977.%E6%9C%89%E5%BA%8F%E6%95%B0%E7%BB%84%E7%9A%84%E5%B9%B3%E6%96%B9.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC)
- 双指针：一个在头，一个在尾，比较两者绝对值大小
- 暴力排序：先平方，再排序

# 滑动窗口
