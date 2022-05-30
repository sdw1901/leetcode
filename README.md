### 1.栈队列堆

#### 9. 1.用两个栈实现队列

```java
package offer;
import java.util.Deque;
import java.util.LinkedList;

/**
 * 剑指 Offer 09. 用两个栈实现队列
 * 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，
 * 分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead操作返回 -1 )
 *
 * 链接：https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof
 *
 * 时间复杂度O(1)
 * 空间复杂度O(N)
 */
public class offer_09_01_CQueue_二个栈实现队列 {
    //进的栈
    Deque<Integer> inStack;
    //出的栈
    Deque<Integer> outStack;

    public offer_09_01_CQueue_二个栈实现队列() {
        inStack = new LinkedList<Integer>();
        outStack = new LinkedList<Integer>();
    }

    public void appendTail(int e) {
        inStack.push(e);
    }

    /**
     * 如果出的栈为空就一次性把inStack的元素放进去
     * 如果出的栈不为空,就正常出。
     * @return
     */
    public int deleteHead() {
        if(outStack.isEmpty()) {
            while (!inStack.isEmpty()) {
                outStack.push(inStack.pop());
            }
        }
        return outStack.isEmpty()?-1:outStack.pop();
    }
}

```



#### 9. 2.用队列实现栈

```java
package offer;
import java.util.LinkedList;
import java.util.Queue;

/**
 * 一个队列实现栈
 */
public class offer_09_02_CStack_队列实现栈 {
    Queue<Integer> queue;

    public offer_09_02_CStack_队列实现栈() {
        queue = new LinkedList<>();
    }

    void push(int x) {
        queue.offer(x);
    }
  
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int size = queue.size() - 1;
        while(size-- > 0) {
            queue.offer(queue.poll());
        }
        return queue.poll();
    }

    /** Get the top element. */
    public int top() {
        return queue.peek();
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
        return queue.isEmpty();
    }
}
```



#### 30.获取栈的最小值

```java
package offer;
import java.util.Deque;
import java.util.LinkedList;

/**
 * 实现 MinStack 类:
 *
 * MinStack() 初始化堆栈对象。
 * void push(int val) 将元素val推入堆栈。
 * void pop() 删除堆栈顶部的元素。
 * int top() 获取堆栈顶部的元素。
 * int getMin() 获取堆栈中的最小元素。
 *
 * 链接：https://leetcode-cn.com/problems/min-stack
 */
public class offer_30_MinStack_获取栈的最小值 {
    Deque<Integer> dataStack;
    Deque<Integer> minStack;
    public offer_30_MinStack_获取栈的最小值() {
        dataStack = new LinkedList<>();
        minStack = new LinkedList<>();
    }

    public void push(int val) {
        dataStack.push(val);
        if (minStack.isEmpty()) {
            minStack.push(val);
        }else {
            //minStack里面有重复的元素保持和dataStack数目一样,pop逻辑更容易一点
            minStack.push(Math.min(val,minStack.peek()));
        }
    }

    public void pop() {
        dataStack.pop();
        minStack.pop();
    }

    public int top() {
        return dataStack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
```



#### 59.1.滑动窗口的最大值

![image-20220412220513532](../pic/滑动窗口的最大值.png)

```java
package offer;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;

/**
 * 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
 * <p>
 * 输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
 * 输出: [3,3,5,5,6,7]
 * 解释:
 * 
 * 滑动窗口的位置                  最大值
 * ---------------               -----
 * [1  3  -1] -3  5  3  6  7       3
 * 1  [3  -1  -3] 5  3  6  7       3
 * 1   3 [-1  -3  5] 3  6  7       5
 * 1   3  -1 [-3  5  3] 6  7       5
 * 1   3  -1  -3 [5  3  6] 7       6
 * 1   3  -1  -3  5 [3  6  7]      7
 * 
 * 链接：https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof
 */
public class offer_59_1_maxInWindow_滑动窗口的最大值 {

    // 单调递减队列实现,第一个是最大值
    // 时间复杂度 O(N)
    // 空间复杂度 O(k)窗口大小
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0 || k == 0) return new int[0];
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> queue = new LinkedList<Integer>();
        for (int i = 0; i < k; i++) {
            //看尾部 因为要一个一个删除  3 1  要加入2    变成 3 2
            while (!queue.isEmpty() && nums[i] > nums[queue.peekLast()]) {
                queue.pollLast();
            }
            //加进去的是索引
            queue.offerLast(i);
        }
        int index = 0;
        res[index++] = nums[queue.peekFirst()];
        for (int i = k; i < nums.length; i++) {
            // 看头索引是否已经过了窗口了
            if (queue.peekFirst() == i - k) {
                queue.pollFirst();
            }
            while (!queue.isEmpty() && nums[i] > nums[queue.peekLast()]) {
                queue.pollLast();
            }
            queue.offerLast(i);
            res[index++] = nums[queue.peekFirst()];
        }
        return res;

    }

    
}
```



#### 59.2.队列的最大值

![image-20220413102844899](../pic/队列的最大值.png)

```java
package offer;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Queue;

/**
 * 队列最大值
 *
 * 使用单调递减队列
 */
public class offer_59_2_queueWithMax_队列的最大值 {
    //正常队列
    Queue<Integer> queue;
    //单调递减队列
    Deque<Integer> decreaseQueue;

    public offer_59_2_queueWithMax_队列的最大值() {
        queue = new LinkedList<Integer>();
        decreaseQueue = new LinkedList<Integer>();
    }

    public int max_value() {
        return decreaseQueue.isEmpty()?-1:decreaseQueue.peekFirst();
    }

    public void push_back(int value) {
        queue.offer(value);
        // 3 1 2
        while(!decreaseQueue.isEmpty() && decreaseQueue.peekLast() < value) {
            decreaseQueue.pollLast();
        }
        decreaseQueue.offerLast(value);
    }

    public int pop_front() {
        if(queue.isEmpty()) return -1;
        //这里是Integer类型
        if(queue.peek().equals(decreaseQueue.peekFirst())) {
            decreaseQueue.pollFirst();
        }
        return queue.poll();
    }
}
```



#### 31.是否是正确的弹栈序列

```java
package offer;

import java.util.Deque;
import java.util.LinkedList;

/**
 * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
 * 假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，
 * 但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。
 * <p>
 * 链接：https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof
 */
public class offer_31_validStackSequence_是否是正确的弹栈序列 {
    public boolean validStackSequence(int[] pushed, int[] popped) {
        Deque<Integer> stack = new LinkedList<>();
        for (int pushedIndex = 0, poppedIndex = 0; pushedIndex < pushed.length; pushedIndex++) {
            stack.push(pushed[pushedIndex]);
            while(!stack.isEmpty() && stack.peek() == popped[poppedIndex])  {
                stack.pop();
                poppedIndex++;
            }
        }
        return stack.isEmpty();
    }

    public static void main(String[] args) {
        int[] pushed = {1,2,3,4,5};
        int[] popped = {4,5,3,2,1};
        boolean result = new offer_31_validStackSequence_是否是正确的弹栈序列().validStackSequence(pushed, popped);
        System.out.println(result);
    }
}

```



#### 40.最小的k个数

```java
package offer;
import java.util.*;

/**
 * 输入整数数组 arr ，找出其中最小的 k 个数。
 * 例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
 */
public class offer_40_getLeastNumbers_获取最小的k个数 {
    //最大堆实现
    //时间复杂度O(nlogk)
    //空间复杂度0(k)
    public List<Integer> getLeastNumbers(int[] arr, int k) {

        Queue<Integer> maxHeap = new PriorityQueue<>((o1, o2) -> (o2 - o1));

        for (int num : arr) {
            maxHeap.offer(num);
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
        }

        return new ArrayList<>(maxHeap);
    }

    //快排思想
    //时间复杂度O(n)
    public int[] getLeastNumbers_partition(int[] arr, int k) {
        int[] res = new int[k];
        int lo = 0, hi = arr.length - 1;
        while (lo < hi) {
            int index = partition(arr, lo, hi);
            // 等于k,[0,k-1]一共k个数
            if (index == k) break;
            else if (index < k) lo = index + 1;
            else hi = index - 1;
        }

        for (int i = 0; i < k; i++) {
            res[i] = arr[i];
        }
        return res;
    }

    public int partition(int[] arr, int lo, int hi) {
        int random = new Random().nextInt(hi-lo+1)+lo;
        swap(arr,random,hi);
        int pivot = arr[hi];
        int less = lo;
        int great = lo;
        for(;great<hi;great++) {
            if(arr[great] < pivot) {
                swap(arr,great,less);
                less++;
            }
        }
        swap(arr,less,hi);
        return less;
    }

    public void swap(int[] arr,int i,int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

}
```



#### 41.数据流中的中位数

```java
package offer;

import java.util.PriorityQueue;
import java.util.Queue;

/**
 * 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
 * 如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
 *
 * 链接：https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof
 *
 * 时间复杂度:查找中位数 O(1) 添加数字 O(log N)
 * 空间复杂度:O(N)
 *
 */
public class offer_41_findMedian_数据流中的中位数 {
    Queue<Integer> maxHeap;
    Queue<Integer> minHeap;
    int count = 0;

    /**
     * initialize your data structure here.
     */
    public offer_41_findMedian_数据流中的中位数() {
        maxHeap = new PriorityQueue<>((o1, o2) -> (o2 - o1));
        minHeap = new PriorityQueue<>();
    }

    public void addNum(int num) {
        if (count % 2 == 0) {
            maxHeap.offer(num);
            // 偶数插入右边,多的情况下右边多
            minHeap.offer(maxHeap.poll());
        } else {
            minHeap.offer(num);
            maxHeap.offer(minHeap.poll());
        }
        count++;
    }

    public double findMedian() {
        if (count % 2 == 0) return (maxHeap.peek() + minHeap.peek()) / 2.0;
        else return minHeap.peek();
    }
}
```



### 2.链表

#### 6.逆序打印链表/24.反转链表

```java
package offer;

import java.util.Deque;
import java.util.LinkedList;

/**
 * 剑指 Offer 06. 从尾到头打印链表
 * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
 * https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/
 */
public class offer_06_reversePrint_逆序打印链表 {
    public static void main(String[] args) {
        ListNode head = new ListNode(1);
        ListNode node1 = new ListNode(3);
        ListNode node2 = new ListNode(2);
        head.next = node1;
        node1.next = node2;
        int[] res = new offer_06_reversePrint_逆序打印链表().reversePrint_V2(head);
        for (int i = 0; i < res.length; i++) {
            System.out.println(res[i]);
        }
    }

    public static class ListNode {
        int val;
        ListNode next;
        public ListNode(int x) {
            val = x;
        }
    }

    /**
     * 方法1,使用双指针翻转链表
     * 时间复杂度O(N)
     * 空间复杂度O(1)
     */
    public int[] reversePrint(ListNode head) {
        ListNode curr = null;
        ListNode pre = head;
        //记录链表长度
        int count = 0;
        while (pre != null) {
            ListNode tmp = pre.next;
            pre.next = curr;
            curr = pre;
            pre = tmp;
            count++;
        }
        int[] res = new int[count];
        // curr就是最后一个指针
        for (int i = 0; i < res.length; i++) {
            res[i] = curr.val;
            curr = curr.next;
        }
        return res;

    }

    /**
     * 逆序用栈
     * 方法2,使用栈 不允许修改链表的时候
     * 时间复杂度O(N)
     * 空间复杂度O(N)
     */
    public int[] reversePrint_V2(ListNode head) {
        Deque<Integer> stack = new LinkedList<>();
        ListNode curr = head;
        while (curr != null) {
            stack.push(curr.val);
            curr = curr.next;
        }
        int[] res = new int[stack.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = stack.pop();
        }
        return res;
    }
}
```



#### 18.1.O(1)时间删除链表的节点

```java
package offer;

/**
 * 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
 * 返回删除后的链表的头节点。
 * 
 * 注意：此题对比原题有改动,原题给的2个指针,可使用O(1)时间复杂度解决
 
 * 输入: head = [4,5,1,9], val = 5
 * 输出: [4,1,9]
 * 解释: 给定你链表中值5的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

 * 链接：https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof
 */
public class offer_18_01_deleteNode_删除链表的节点 {

    public static class ListNode {
        int val;
        ListNode next;
        public ListNode(int x) {
            val = x;
        }
    }

    // 时间复杂度 O(n)
    // 空间复杂度 O(1)
    public ListNode deleteNode(ListNode head, int val) {
        // 使用虚拟节点,最后返回头结点的时候使用
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode curr = dummy;
        while (curr.next != null) {
            if (curr.next.val == val) {
                curr.next = curr.next.next;
            }
            curr = curr.next;
        }
        return dummy.next;
    }


    // 时间复杂度 O(1)
    // 空间复杂度 O(1)
    public ListNode deleteNode2(ListNode head, ListNode deleteNode) {
        if (head == null || deleteNode == null || deleteNode == head) return null;
        //非尾巴节点
        if (deleteNode.next != null) {
            ListNode next = deleteNode.next;
            deleteNode.val = next.val;
            deleteNode.next = next.next;
            //尾巴节点从头遍历
        } else {
            ListNode curr = head;
            while (curr.next != deleteNode) {
                curr = curr.next;
            }
            curr.next = deleteNode.next;
        }
        return head;

    }
}
```



#### 18.2.删除链表中重复的节点

```java
package offer;

/**
 * 给定一个已排序的链表的头 head ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表 。
 */
public class offer_18_02_deleteDuplication_删除链表重复的节点 {
    public static class ListNode {
        int val;
        ListNode next;

        public ListNode(int x) {
            val = x;
        }
    }

    // 时间复杂度 O(n)
    // 空间复杂度 O(1)
    public ListNode deleteDuplication(ListNode head) {
        // 不使用虚拟节点也可以,head节点不会被删,最后返回头结点的时候使用
//        ListNode dummy = new ListNode(-1);
//        dummy.next = head;
//        ListNode curr = dummy;
        if(head == null) return null;
        ListNode curr = head;
        while(curr.next!=null) {
            if(curr.val == curr.next.val) {
                curr.next = curr.next.next;
            }else {
                curr = curr.next;
            }
        }
//        return dummy.next;
        return head;
    }
}
```



#### 22.链表中倒数第 K 个结点

```java
package offer;

/**
 * 倒数第k个节点
 */
public class offer_22_getKthFromEnd_链表的倒数第k个节点 {

    public static class ListNode {
        int val;
        ListNode next;

        public ListNode(int val) {
            this.val = val;
        }
    }

    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode pre = head;
        ListNode curr = head;
        for (int i = 0; i < k; i++) {
            //加判空
            if(pre == null) return null;
            pre = pre.next;
        }
        while (pre != null) {
            pre = pre.next;
            curr = curr.next;
        }
        return curr;
    }

```



#### 23.链表中环的入口结点

```java
package offer;

/**
 * 1.是否有环
 * 2.找到入环节点
 */
public class offer_23_hasCycle_环形链表 {
    public static class ListNode {
        int val;
        ListNode next;
        public ListNode(int val) {
            this.val = val;
        }
    }

    // 时间复杂度 O(n)
    // 空间复杂度 O(1)
    public boolean hasCycle(ListNode head) {
        ListNode pFast = head;
        ListNode pSlow = head;
        while (pFast != null && pFast.next != null) {
            pFast = pFast.next.next;
            pSlow = pSlow.next;
            if (pFast == pSlow) return true;
        }
        return false;
    }

    public ListNode entryNodeOfLoop(ListNode head) {
        ListNode pFast = head;
        ListNode pSlow = head;
        while (pFast != null && pFast.next != null) {
            pFast = pFast.next.next;
            pSlow = pSlow.next;
            if (pFast == pSlow) {
                pFast = head;
                while (pFast != pSlow) {
                    pFast = pFast.next;
                    pSlow = pSlow.next;
                }
                return pFast;
            }
        }
        return null;
    }
}
```



#### 25.合并两个排序的链表

```java
package offer;

/**
 * 合并有序链表
 */
public class offer_25_merge_合并二个有序的链表 {
    public static class ListNode {
        int val;
        ListNode next;
        public ListNode(int val) {
            this.val = val;
        }
    }

    // 时间复杂度 O(n)
    // 空间复杂度 O(1)
    public ListNode merge(ListNode l1,ListNode l2) {
        //最后返回的时候使用
        ListNode dummy = new ListNode(-1);
        ListNode curr = dummy;
        while(l1!=null&&l2!=null) {
            if(l1.val<=l2.val) {
                curr.next = l1;
                l1 = l1.next;
            }else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }
        curr.next = l1==null?l2:l1;
        return dummy.next;
    }

    /**
     * 递归
     * @param l1
     * @param l2
     * @return
     */
    public ListNode merge2(ListNode l1,ListNode l2) {
        if(l1==null) return l2;
        if(l2==null) return l1;
        if(l1.val<=l2.val) {
            l1.next = merge2(l1.next,l2);
            return l1;
        }else {
            l2.next = merge2(l1,l2.next);
            return l2;
        }

    }
}
```



#### 35.复杂链表的复制

```java
package offer;

import java.util.HashMap;

/**
 * 请实现 copyRandomList 函数，复制一个复杂链表。
 * 在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
 * <p>
 * 链接：https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof
 */
public class offer_35_cloneComplexNode_复杂链表的复制 {

    public class ComplexNode {
        int val;
        ComplexNode next;
        ComplexNode random;

        public ComplexNode(int x) {
            val = x;
        }
    }

    // 时间复杂度 O(N)
    // 空间复杂度 O(N)
    public ComplexNode cloneComplexNode(ComplexNode head) {
        if (head == null) return null;
        HashMap<ComplexNode, ComplexNode> map = new HashMap<>();
        ComplexNode curr = head;
        //复制节点
        while (curr != null) {
            map.put(curr, new ComplexNode(curr.val));
            curr = curr.next;
        }
        //复制链接
        curr = head;
        while (curr != null) {
            map.get(curr).next = map.get(curr.next);
            map.get(curr).random = map.get(curr.random);
            curr = curr.next;
        }
        //返回新链表的头节点
        return map.get(head);
    }
}
```



#### 52.两个链表的第一个公共结点

```java
package offer;

/**
 * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。
 * 如果两个链表不存在相交节点，返回 null 。
 */
public class offer_52_gerIntersection_找出链表相交起始点 {

    public static class ListNode{
        int val;
        ListNode next;
        public ListNode(int x) {
            val = x;
        }
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null) {
            return null;
        }
        ListNode tmpA = headA;
        ListNode tmpB = headB;

        while(tmpA != tmpB) {
            tmpA = tmpA==null?headB:tmpA.next;
            tmpB = tmpB==null?headA:tmpB.next;

        }
        return tmpA;
    }
}
```



#### 回文链表

<img src="/Users/chenjing/Library/Application Support/typora-user-images/image-20220414090733885.png" alt="image-20220414090733885" style="zoom:50%;" />

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        if(head==null||head.next==null) return true;
        ListNode fast = head;
        ListNode slow = head;
        //poster用来断掉前半段与后半段的指针
        ListNode poster = null;
        while(fast!=null&&fast.next!=null) {
            fast = fast.next.next;
            //poster保持为slow的前一个
            poster = slow;
            slow = slow.next;
        }
        if(fast==null) {
            //偶数个
            poster.next = null;
            slow = reverse(slow);
        }else{
            //奇数个
            poster.next = null;
            ListNode temp = slow.next;
            slow.next = null;
            slow = reverse(temp);
        }
        while(head!=null) {
            if(head.val != slow.val) return false;
            slow = slow.next;
            head = head.next;
        }
        return true;

    }

    public ListNode reverse(ListNode head) {
        ListNode curr = null;
        ListNode pre = head;
        while(pre!=null) {
            ListNode next = pre.next;
            pre.next = curr;
            curr = pre;
            pre = next;
        }
        return curr;
    }
```







#### 回文字符串

![image-20220328004624990](/Users/chenjing/Library/Application Support/typora-user-images/image-20220328004624990.png)



#### 链表相加

```java
/**
* leetcode445
* 给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。
* 它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。
* 你可以假设除了数字 0 之外，这两个数字都不会以零开头。
* 链接：https://leetcode-cn.com/problems/add-two-numbers-ii
*/
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode l1R = reverse(l1);
        ListNode l2R = reverse(l2);
        ListNode dummy = new ListNode(-1);
        ListNode curr = dummy;
        int reminder = 0;
        while(l1R!=null||l2R!=null||reminder!=0) {
            int a = l1R==null?0:l1R.val;
            int b = l2R==null?0:l2R.val;
            int sum = a + b + reminder;
            //余数
            reminder = sum/10;
            sum = sum%10;
            curr.next = new ListNode(sum);
            curr = curr.next;
            //防止空指针
            l1R = l1R==null?l1R:l1R.next;
            l2R = l2R==null?l2R:l2R.next;
        }
        return reverse(dummy.next);
    }


    public ListNode reverse(ListNode head) {
        ListNode pre = head;
        ListNode curr = null;
        while(pre!=null) {
            ListNode next = pre.next;
            pre.next = curr;
            curr = pre;
            pre = next;
        }
        return curr;

    }
}
```





### 3.树

#### 7.重建二叉树

```java
package offer;

import java.util.HashMap;

/**
 * 剑指 Offer 07. 重建二叉树
 * 输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。
 * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
 * https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/
 * preLeft                                         preLeft+index-inLeft+1         preRight
 *|   根    |             左                       |                         右            |
 *           preLeft+1         preLeft+index-inLeft
 *
 *                                     index
 *|            左                     |  根  |                 右                          |
 * inLeft                     index-1        index+1                               inRight
 */
public class offer_07_buildTree_重建二叉树 {
    //存放中序遍历元素的索引位置,前序遍历的第一个元素就是根节点,要找到中序遍历中该根节点的索引
    static HashMap<Integer, Integer> indexHash = new HashMap<Integer, Integer>();

    public static void main(String[] args) {
        int[] preOrder = {3, 9, 20, 15, 7};
        int[] inOrder = {9, 3, 15, 20, 7};
        for (int i = 0; i < inOrder.length; i++) {
            indexHash.put(inOrder[i], i);
        }
        TreeNode res = new offer_07_buildTree_重建二叉树().buildTree(preOrder, 0, preOrder.length - 1,
                inOrder, 0, inOrder.length - 1);
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int value) {
            val = value;
        }
    }

    // 时间复杂度O(N)
    // 空间复杂度O(N)
    public TreeNode buildTree(int[] preOrder, int preLeft, int preRight,
                              int[] inOrder, int inLeft, int inRight) {
        if (preLeft > preRight || inLeft > inRight) {
            return null;
        }
        //获取根节点索引
        int rootVal = preOrder[preLeft];
        Integer index = indexHash.get(rootVal);

        TreeNode root = new TreeNode(rootVal);
        root.left = buildTree(preOrder, preLeft + 1, preLeft + index - inLeft, inOrder, inLeft, index - 1);
        root.right = buildTree(preOrder, preLeft + index - inLeft + 1, preRight, inOrder, index + 1, inRight);
        return root;
    }
}
```



#### 8.二叉树的下一个结点

```java
package offer;

/**
 * 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
 *
 * 思路：此题包含三步：
 * 1.如果此节点有右子树，下一个节点为右子节点的最左边的节点。
 * 2.如果此节点没有右子树，并且如果此节点是其父节点的左子节点，则下一个节点为父节点。
 * 3.如果此节点没有右子树，并且如果此节点是其父节点的右子节点，则一直向上找，直到找到第一个是其父节点左节点的节点，下一个节点就为此节点。
 */
public class offer_08_getNext_二叉树的下一个节点 {

    public class TreeLinkNode {
        int val;
        TreeLinkNode left;
        TreeLinkNode right;
        //指向父节点的指针
        TreeLinkNode next;

        public TreeLinkNode(int value) {
            val = value;
        }
    }

    /**
     * 时间复杂度：O(N)
     * 空间复杂度：O(1)
     * @param pNode
     * @return
     */
    public TreeLinkNode getNext(TreeLinkNode pNode) {
        //有右子树
        if(pNode.right!=null) {
            TreeLinkNode right = pNode.right;
            while(right.left!=null) {
                right = right.left;
            }
            return right;
        //无右子树
        }else {
            while(pNode.next!=null) {
                TreeLinkNode parent = pNode.next;
                // 2和3同一个判断条件
                if (parent.left == pNode) {
                    return parent;
                }
                pNode = pNode.next;
            }
        }
        return null;
    }
}
```



#### 26.B是否是A的子树

```java
package offer;

/**
 * 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
 *
 * B是A的子结构， 即 A中有出现和B相同的结构和节点值。
 * 例如:
 * 给定的树 A:
 *
 *            3
 *           / \
 *          4   5
 *         / \
 *        1   2
 * 给定的树 B：
 *
 *       4
 *      /
 *     1
 * 返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。
 *
 * 链接：https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof
 *
 * 时间复杂度 O(MN)
 * 空间复杂度 O(M)
 */
public class offer_26_hasSubTree_B是否是A的子树 {

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int value) {
            val = value;
        }
    }


    public boolean hasSubTree(TreeNode A, TreeNode B) {
        //第一个递归的出口
        if (A == null || B == null) return false;
        //一个个看 或 找到就行
        return helper(A, B) || hasSubTree(A.left, B) || hasSubTree(A.right, B);
    }

    //找到根节点后判断是否是子树
    public boolean helper(TreeNode pA, TreeNode pB) {
        // 第二个递归的出口
        if (pB == null) return true;
        if (pA == null) return false;
        if (pA.val != pB.val) return false;
        // 与,都满足才行
        return helper(pA.left, pB.left) && helper(pA.right, pB.right);
    }
}
```



#### 27.输出二叉树的镜像

```java
package offer;

/**
 * 输出二叉树的镜像
 *
 * https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/
 */
public class offer_27_returnMirror_输出二叉树的镜像 {

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int x) {
            val = x;
        }
    }

    // 时间复杂度 O(N)
    // 空间复杂度 O(N)
    public TreeNode mirrorTree(TreeNode root) {
        // 特殊情况
        if (root == null) return null;
        if (root.left == null && root.left == null) return root;
        swap(root);
        mirrorTree(root.left);
        mirrorTree(root.right);
        return root;
    }

    public void swap (TreeNode root) {
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
    }
}
```



#### 28.是否是对称的二叉树

```java
package offer;

/**
 * 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
 *
 * 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
 */
public class offer_28_isSymmetric_是否是对称二叉树 {

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    // 时间复杂度 O(N)
    // 空间复杂度 O(N)
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isMirror(root.left, root.right);
    }

    public boolean isMirror(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        return isMirror(left.left, right.right) && isMirror(left.right, right.left);
    }
  
}
```



#### 32.层序遍历二叉树

```java
package offer;

import java.util.*;

/**
 * 层序遍历
 */
public class offer_32_levelTraversal_层序遍历 {

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int x) {
            val = x;
        }
    }

    // 时间复杂度 O(N)
    // 空间复杂度 O(N)
    public List<List<Integer>> levelTraversal(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode current = queue.poll();
                list.add(current.val);
                if (current.left != null) queue.offer(current.left);
                if (current.right != null) queue.offer(current.right);
            }
            res.add(list);
        }
        return res;
    }

    //之字形打印
    public List<List<Integer>> levelTraversal2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int layerIndex = 1;
        while (!queue.isEmpty()) {
            List<Integer> layer = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode current = queue.poll();
                layer.add(current.val);
                if (current.left != null) queue.offer(current.left);
                if (current.right != null) queue.offer(current.right);
            }
            if (layerIndex % 2 == 0) {
//                Collections.reverse(layer);
                for (int i = 0, j = layer.size() - 1; i < j; i++, j--) {
                    Integer temp = layer.get(i);
                    layer.set(i, layer.get(j));
                    layer.set(j, temp);
                }

            }
            res.add(layer);
            layerIndex++;
        }
        return res;
    }
}
```



#### 33.是否是二叉搜索树的后序遍历

```java
package offer;

/**
 * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。
 * 假设输入的数组的任意两个数字都互不相同。
 *
 * 二叉搜索树的左子树小于根节点,右子树大于根节点。
 * 子递归也一样
 *  左  右  根
 * 小于 大于 等于
 */
public class offer_33_verifySquenceOfBST_是否是二叉搜索树的后序遍历 {
    public boolean verifyPostorder(int[] postorder) {
        if(postorder == null || postorder.length == 0) return false;
        return verifyPostorderBST(postorder,0,postorder.length-1);
    }

    // 时间复杂度 O(N^2)
    // 空间复杂度 O(N)
    public boolean verifyPostorderBST(int[] postorder,int left,int right) {
        if(left>=right) return true;
        int rootVal = postorder[right];
        //先遍历到右子树的第一个节点
        int i = 0;
        for(;i <right;i++) {
            if(postorder[i] > rootVal) {
                break;
            }
        }
        //遍历到最后
        int j = i;
        for(;j<right;j++) {
            if(postorder[j] < rootVal) {
                return false;
            }
        }
        return verifyPostorderBST(postorder,left,i-1)
                &&verifyPostorderBST(postorder,i,right-1);

    }
}
```



#### 34.二叉树路径的和_回溯

```java
package offer;

import java.util.ArrayList;
import java.util.List;

/**
 * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
 *
 * 叶子节点 是指没有子节点的节点。
 *
 *
 * 链接：https://leetcode-cn.com/problems/path-sum-ii
 */
public class offer_34_findPath_二叉树路径的和_回溯 {

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int x) {
            val = x;
        }
    }

    public List<List<Integer>> findPath(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backTracking(root, targetSum, res, path);
        return res;
    }

    //回溯法
    public void backTracking(TreeNode root, int target, List<List<Integer>> res, List<Integer> path) {
        //递归结束条件
        if(root==null) return;
        path.add(root.val);
        target = target - root.val;
        // 满足条件 target == 0 且是叶子节点
        if(target == 0&&root.left==null&&root.right==null) {
            res.add(new ArrayList<>(path));
        }
        backTracking(root.left,target,res,path);
        backTracking(root.right,target,res,path);
        path.remove(path.size()-1);
    }
}
```



#### 36.二叉搜索树转排序双向链表

```java
package offer;

import java.util.ArrayList;
import java.util.List;

/**
 * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。
 * 要求不能创建任何新的节点，只能调整树中节点指针的指向。
 * <p>
 * https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/
 */
public class offer_36_treeToDoublyList_二叉搜索树转排序双向链表 {

    public static class Node {
        int val;
        Node left;
        Node right;

        public Node(int x) {
            val = x;
        }

    }

    // 时间复杂度 O(N)
    // 空间复杂度 O(N)
    public Node treeToDoublyList(Node root) {
        if(root == null) return null;
        List<Node> list = new ArrayList<>();
        //中序遍历拿到排序好的节点
        dfs(root, list);
        //先处理头尾节点特殊情况
        list.get(0).left = null;
        list.get(list.size() - 1).right = null;
        for (int i = 0; i < list.size(); i++) {
            // i-1>=0
            if(i>=1) {
                list.get(i).left = list.get(i-1);
            }
            // i-1<list.size()
            if(i<list.size()-1) {
                list.get(i).right = list.get(i+1);
            }
        }
        return list.get(0);
    }

    public void dfs(Node root, List<Node> list) {
        if (root == null) return;
        dfs(root.left, list);
        list.add(root);
        dfs(root.right, list);
    }
}

```



#### 54.二叉搜索树第k大的值

```java
package offer;

/**
 * 给定一棵二叉搜索树，请找出其中第 k 大的节点的值。
 * 倒数k个:反中序遍历
 * 第k个: 中序遍历
 */
public class offer_54_kthLargest_二叉搜索树第k大的值 {

    public class TreeNode{
        int val;
        TreeNode left;
        TreeNode right;
        public TreeNode(int x) {
            val = x;
        }
    }

    int res = -1,k;
    // 第k大,倒数,反中序遍历
    public int kthLargest(TreeNode root,int k) {
        this.k = k;
        dfs(root);
        return res;
    }

    public void dfs(TreeNode root) {
        if(root == null) return;
        dfs(root.right);
        if(--k == 0) {
            res = root.val;
            return;
        }
        dfs(root.left);
    }

}
```



#### 55.1.二叉树的深度

```java
package offer;

import java.util.LinkedList;
import java.util.Queue;

/**
 * 二叉树的深度
 */
public class offer_55_1_treeDepth_二叉树的深度 {
    public static class TreeNode{
        int val;
        TreeNode left;
        TreeNode right;
        public TreeNode(int x) {
            val = x;
        }
    }

    //时间复杂度O(n) 每个节点都会遍历到
    //空间复杂度O(n)
    public int maxDepth(TreeNode root) {
        int depth = 0;
        if(root==null) return depth;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            int size = queue.size();
            depth++;
            for(int i = 0;i<size;i++) {
                TreeNode curr = queue.poll();
                if(curr.left!=null) queue.offer(curr.left);
                if(curr.right!=null) queue.offer(curr.right);
            }

        }
        return depth;
    }

    //时间复杂度O(n) 每个节点都会遍历到
    //空间复杂度O(height)
    public int maxDepth_recur(TreeNode root) {
        if(root==null) return 0;
        int left = maxDepth_recur(root.left);
        int right = maxDepth_recur(root.right);
        return Math.max(left,right)+1;
    }

}
```



#### 55.2.平衡二叉树

```java
package offer;

/**
 * 是否是平衡二叉树
 */
public class offer_55_2_balanceTree_是否是平衡二叉树 {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int x) {
            val = x;
        }
    }

    private boolean res = true;

    public boolean isBalanced(TreeNode root) {
        dfs(root);
        return res;
    }

    public int dfs(TreeNode root) {
        if (root == null) return 0;
        int left = dfs(root.left);
        int right = dfs(root.right);
        if (Math.abs(left - right) > 1) {
            res = false;
        }
        return Math.max(left, right) + 1;
    }
}
```



#### 68.树中两个节点的最低公共祖先

```java
package offer;

/**
 * 二叉树的最低公共祖先
 */
public class offer_68_lowestCommonAncestor_二叉树的最低公共祖先 {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        public TreeNode(int x) {
            val = x;
        }
    }

    // 时间复杂度O(n)
    // 空间复杂度O(n)
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 先看根节点
        if(root == null || root.val == p.val || root.val == q.val) return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left!=null&&right!=null) return root;
        else if(left == null) return right;
        else if(right == null) return left;
        else return null;
    }

    // 如果是二叉搜索树
    // 时间复杂度O(n)
    // 空间复杂度O(n)
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        while(root != null) {
            if(root.val < p.val && root.val < q.val) // p,q 都在 root 的右子树中
                root = root.right; // 遍历至右子节点
            else if(root.val > p.val && root.val > q.val) // p,q 都在 root 的左子树中
                root = root.left; // 遍历至左子节点
            else break;
        }
        return root;
    }
}
```



### 4.数学与位运算

#### 15.二进制中1的个数

```java
package offer;

/**
 * 编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。
 *
 * 输入：n = 11 (控制台输入 00000000000000000000000000001011)
 * 输出：3
 * 解释：输入的二进制串 00000000000000000000000000001011中，共有三位为 '1'。
 *
 * 链接：https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof
 */
public class offer_15_hammingWeight_二进制中1的个数 {
    // 时间复杂度 O(logn) 移动是/2操作
    // 空间复杂度 O(1)
    public int hammingWeight(int n) {
        int res = 0;
        while(n!=0) {
            res += n&1;
            n >>>= 1;
        }
        return res;
    }

    // 时间复杂度 O(M)
    // 空间复杂度 O(1)
    //n&(n−1) 去掉最后一位
    public int hammingWeight_V2(int n) {
        int res = 0;
        while(n != 0) {
            res++;
            n &= n - 1;
        }
        return res;
    }
}
```



#### 56.数组中只出现一次的数字

```java
package offer;

/**
 * 利用以下特性
 * a ^ a = 0
 * a ^ 0 = a
 * a ^ b ^ c = a ^ c ^ b
 * <p>
 * 1.其他出现2次，只有1个出现1次。{4,1,2,1,2}
 */
public class offer_56_onlyOnce_数组中只出现一次的数字 {

    /**
     * 1.其他出现2次，只有1个出现1次。{4,1,2,1,2}
     */
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            res ^= nums[i];
        }
        return res;
    }

    /**
     * 1.其他出现3次，只有1个出现1次。{3，2，3，3}
     */
    public int singleNumber2(int[] nums) {
        int res = 0;

        // int有32位
        for (int i = 0; i < 32; i++) {
            int oneCount = 0;
            //统计每一位数上的1的个数 num >> i & 1 第i位是1就是1不是1就是0
            for (int num : nums) {
                oneCount += (num >> i) & 1;
            }
            //被3整除 那一位就是0
            //不被3整除 那一位就是1
            //在第i位加1就或上 1 << i
            if (oneCount % 3 != 0) {
                res |= 1 << i;
            }

        }
        return res;
    }

    /**
     * 其中恰好有两个元素只出现一次，其余所有元素均出现两次。
     * 利用除答案以外的元素均出现两次，我们可以先对 nums 中的所有元素执行异或操作，
     * 得到 sum , 为两答案的异或值（sum必然不为 0）。
     *
     * 然后取sum二进制表示中为1的任意一位 k,sum 中的第 k 位为 1 意味着两答案的第 k 位二进制表示不同。
     *
     * 对nums进行遍历，对第k位分别为0和1的元素分别求异或和（两答案必然会被分到不同的组），即为答案。

     */
    public int[] singleNumber3(int[] nums) {
        int sum = 0;
        for (int i : nums) sum ^= i;
        int k = -1;
        for (int i = 31; i >= 0 && k == -1; i--) {
            if (((sum >> i) & 1) == 1) k = i;
        }
        int[] ans = new int[2];
        for (int i : nums) {
            if (((i >> k) & 1) == 1) ans[1] ^= i;
            else ans[0] ^= i;
        }
        return ans;
    }
}
```



#### 64.不用乘除计算1_n的和

```java
package offer;

/**
 * 不用乘除计算1+2+3+...+n
 *
 * 使用&&来递归
 */
public class offer_64_sumNums_不用乘除计算1_n的和 {
    private int res = 0;
    public int sumNums(int n) {
        // n>1才执行后面的递归
        boolean flag = n > 1 && sumNums(n-1) > 0;
        res += n;
        return res;
    }
}

```



#### 65.不用加减乘除实现加法

```java
package offer;

/**
 * 求两正数之和 不用加减乘除
 */
public class offer_65_add_不用加减乘除实现加法 {

    public int add(int a, int b) {
        // 不进位和进位相加又进入循环
        while(b != 0) { // 当进位为 0 时跳出
            // c = 进位
            int c = (a & b) << 1;
            a ^= b; // a = 非进位和
            b = c; // b = 进位
        }
        return a;
    }

}
```



#### 66.1-n中1出现的个数

```java
package offer;

/**
 * 给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。
 */
public class offer_43_numberOf1_1出现的个数 {

    //时间复杂度O(nlogn)
    public int numberOf1(int n) {
        int count = 0;
        for(int i = 1;i<=n;i++) {
            while(i!=0) {
                if(i%10 == 1) {
                    count++;
                }
                i/=10;
            }
        }
        return count;
    }

    //个十百千.. 1的个数
  	//我们以n=2021为例，所有小于等于 2021 的数中个位一共会出现多少个 1 呢？
		//我们可以很容易地发现，个位数出现1的频率是每10个数出现一次，对不对？
		//所以，个位数出现多少 1 就取决于，一个有多少个 10，比如 2021 一共用 202 个 10，
    //所以，个位出现 1 的数一共有 202 次（1， 11， 21，2011）+ 1次（2021）。
    public int countDigitOne(int n) {
        // 2021
        int ans = 0;
        for (int i = 1; i <= n; i *= 10) {
            ans += (n / (i * 10)) * i + Math.min(Math.max(n % (i * 10) - i + 1,0), i);
        }
        return ans;
    }

}

```

### 5.回溯

#### 34.二叉树路径的和_回溯

```java
package offer;

import java.util.ArrayList;
import java.util.List;

/**
 * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
 *
 * 叶子节点 是指没有子节点的节点。
 *
 *
 * 链接：https://leetcode-cn.com/problems/path-sum-ii
 */
public class offer_34_findPath_二叉树路径的和_回溯 {

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        public TreeNode(int x) {
            val = x;
        }
    }

    public List<List<Integer>> findPath(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backTracking(root, targetSum, res, path);
        return res;
    }

    //回溯法
    public void backTracking(TreeNode root, int target, List<List<Integer>> res, List<Integer> path) {
        //递归结束条件
        if(root==null) return;
        path.add(root.val);
        target = target - root.val;
        // 满足条件 target == 0 且是叶子节点
        if(target == 0&&root.left==null&&root.right==null) {
            res.add(new ArrayList<>(path));
        }
        backTracking(root.left,target,res,path);
        backTracking(root.right,target,res,path);
        path.remove(path.size()-1);
    }
}
```



#### 12.矩阵中的路径_回溯

```java
package offer;

/**
 * 矩阵中的路径
 * 请设计一个函数，用来判断在一个n乘m的矩阵中是否存在一条包含某长度为len的字符串所有字符的路径。
 * 路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
 * 如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。
 *
 *  矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，
 *  因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
 *
 *  https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/
 */
public class offer_12_exist_矩阵中的路径_回溯 {
    int[] xBias = {0, 0, 1, -1};
    int[] yBias = {1, -1, 0, 0};

    public boolean exist(char[][] board, String word) {
        char[] target = word.toCharArray();
        //等于word首单词
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                    if (backTracking(board, i, j, target, 0)) return true;
            }
        }
        return false;
    }

    public boolean backTracking(char[][] board, int i, int j, char[] target, int current) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || target[current] != board[i][j]) return false;
        if (current == target.length - 1) return true;

        board[i][j] = '0';

        boolean res = false;
        for (int k = 0; k < xBias.length; k++) {
            if (backTracking(board, i + yBias[k], j + xBias[k], target, current + 1)) {
                res = true;
                //只有递归过程完成目标才会true,所以有true就结束
                break;
            }
        }
        //复原,防止exist中for循环第二次进来的时候结构被改变。
        board[i][j] = target[current];
        return res;
    }

    public static void main(String[] args) {
        boolean res = new offer_12_exist_矩阵中的路径_回溯().exist(new char[][]{{'a'}}, "ab");
        System.out.println(res);
    }
}
```



#### 组合

```java
package backTracking;

import java.util.ArrayList;
import java.util.List;

/**
 * 剑指 Offer II 080. 含有 k 个元素的组合
 * 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
 * https://leetcode-cn.com/problems/uUsW3B/
 */
public class offer_80_combine {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> combine(int n, int k) {
        backTracking(n, k, 1);
        return res;
    }

    public void backTracking(int n, int k, int startIndex) {
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = startIndex; i <= n-(k-path.size())+1; i++) {
            path.add(i);
            backTracking(n, k, i + 1);
            path.remove(path.size() - 1);
        }
    }

    public static void main(String[] args) {
        List<List<Integer>> combine = new offer_80_combine().combine(4, 2);
        System.out.println(1);
    }
}
```

#### 组合之和

```java
package backTracking;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

/**
 * 剑指 Offer II 081. 允许重复选择元素的组合之和
 * 给定一个无重复元素的正整数数组candidates和一个正整数target，找出candidates中所有可以使数字和为目标数target的唯一组合。
 * <p>
 * candidates中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是不同的。
 * <p>
 * 链接：https://leetcode-cn.com/problems/Ygoe9J
 */
public class offer_81_combinationSum {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // startIndex是为了有了12 不再要21了
        // 全排列无startIndex
        backTracking(candidates, target, 0);
        return res;
    }

    public void backTracking(int[] candidates, int target, int startIndex) {
        if (target < 0) return;
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = startIndex; i < candidates.length; i++) {
            path.add(candidates[i]);
            // i和i+1 i+1是防止选自己,不能重复,i代表可以重复。组合按顺序可以
            // 排列里面只能用used[i]
            backTracking(candidates, target - candidates[i], i);
            path.remove(path.size() - 1);
        }
    }

    public static void main(String[] args) {
        new offer_81_combinationSum().combinationSum(new int[]{2, 3, 6, 7}, 7);
    }

}
```



#### 全排列无重复

```java
package backTracking;

import java.util.ArrayList;
import java.util.List;

/**
 * 全排列无重复
 */
public class leetcode_46_permute {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    //排列只能用这个不要重复的 112
    boolean[] used;

    public List<List<Integer>> permute(int[] nums) {
        used = new boolean[nums.length];
        backTracking(nums);
        return res;
    }

    public void backTracking(int[] nums) {
        if(path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i = 0;i<nums.length;i++) {
            if(used[i]) continue;
            used[i] = true;
            path.add(nums[i]);
            backTracking(nums);
            path.remove(nums[i]);
            used[i] = false;
        }
    }
}
```



#### 38.有重复的字符串全排列_回溯

```java
package offer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 全排列去重
 */
public class offer_38_permuteUnique_有重复的字符串全排列_回溯 {
//    List<List<Integer>> res = new ArrayList<>();
//    List<Integer> path = new ArrayList<>();
//    boolean[] used;

//    public List<List<Integer>> permuteUnique(int[] nums) {
//
//        if (nums.length == 0) return res;
//        used = new boolean[nums.length];
//        Arrays.sort(nums);
//        backTracking(nums);
//        return res;
//    }
//
//
//    public void backTracking(int[] nums) {
//        if (path.size() == nums.length) {
//            res.add(new ArrayList<>(path));
//            return;
//        }
//        for (int i = 0; i < nums.length; i++) {
//            if (used[i]) continue;
//            // 如果nums[i] == nums[i-1]说明同层有重复元素
//            // used[i-1] == false 同层有重复元素且遍历完了 可以剪枝
//            // used[i-1] = true 同树枝有重复元素 不能剪枝
//            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
//            used[i] = true;
//            path.add(nums[i]);
//            backTracking(nums);
//            path.remove(path.size() - 1);
//            used[i] = false;
//        }
//    }

    List<String> res = new ArrayList<>();

    boolean[] used;

    public List<String> permuteUnique(String s) {
        if (s.length() == 0) return res;
        used = new boolean[s.length()];
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        backTracking(chars,new StringBuilder());
        return res;
    }

    public String[] permutation_string(String s) {
        used = new boolean[s.length()];
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        backTracking(chars,new StringBuilder());
        return res.toArray(new String[res.size()]);
    }


    public void backTracking(char[] chars,StringBuilder stringBuilder) {
        if (stringBuilder.length() == chars.length) {
            res.add(stringBuilder.toString());
            return;
        }
        for (int i = 0; i < chars.length; i++) {
            if (used[i]) continue;
            // 如果nums[i] == nums[i-1]说明同层有重复元素
            // used[i-1] == false 同层有重复元素且遍历完了 可以剪枝
            // used[i-1] = true 同树枝有重复元素 不能剪枝
            if (i > 0 && chars[i] == chars[i - 1] && !used[i - 1]) continue;
            used[i] = true;
            stringBuilder.append(chars[i]);
            backTracking(chars,stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            used[i] = false;
        }
    }

    public static void main(String[] args) {
//        int[] arr = {1, 1, 2};
        String s = "abc";
        new offer_38_permuteUnique_有重复的字符串全排列_回溯().permuteUnique(s);

    }
}
```





### 6.数组

#### 3.找出数组中重复的元素

```java
package offer;

import java.util.HashSet;

/**
 * 剑指 Offer 03. 数组中重复的数字
 找出数组中重复的数字。
 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。
 数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
 链接：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof

 解题关键：把数字放到指定索引位置,因为数字的范围在长度内

 *
 */
public class offer_03_findRepeatNumber_找出数组中重复的元素 {
    public static void main(String[] args) {
        int[] arr = {2, 3, 1, 0, 2, 5, 3};
        int res = new offer_03_findRepeatNumber_找出数组中重复的元素().findRepeatNumber_V2(arr);
        System.out.println(res);
    }

    // 时间复杂度O(N)
    // 空间复杂度O(N)
    public int findRepeatNumber(int[] nums) {
        HashSet<Integer> checkSet = new HashSet<Integer>();
        for(int i :nums) {
            if(checkSet.contains(i)) {
                return i;
            }
            checkSet.add(i);
        }
        return -1;
    }

    // 时间复杂度O(N)
    // 空间复杂度O(1)
    public int findRepeatNumber_V2(int[] nums) {
        if(nums == null || nums.length == 0) return -1;
        int n = nums.length;
        int i = 0;
        while(i < n) {
            //比较元素和下标
            if(nums[i] != i) {
                //不等于,且要换的位置已经有正确的元素了
                //nums[i]当前位置元素
                //nums[nums[i]]要交换的位置的元素
                if(nums[i] == nums[nums[i]]) {
                    return nums[i];
                }
                //交换
                int tmp = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = tmp;
            }else {
                i++;
            }
        }
        return -1;
    }
  	
  	//不移动数组的元素 二分法
  	//时间复杂度O(nlogn)
  	//空间复杂度O(1)
}
```



#### 4.二维递增数组查找数字

```java
package offer;

/**
 * 剑指 Offer 04. 二维数组中的查找
 * https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/
 * 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
 * 请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
 */
public class offer_04_findNumberIn2DArray_二维递增数组查找数字 {
    public static void main(String[] args) {
        int[][] arr = {
                {1, 4, 7, 11, 15},
                {2, 5, 8, 12, 19},
                {3, 6, 9, 16, 22},
                {10, 13, 14, 17, 24},
                {18, 21, 23, 26, 30}
        };
        int target = 5;
        boolean res = new offer_04_findNumberIn2DArray_二维递增数组查找数字().findNumberIn2DArray(arr, target);
        System.out.println(res);
    }

    // 时间复杂度O(M+N)
    // 空间复杂度O(1)
    public boolean findNumberIn2DArray(int[][] arr, int target) {
        // 因为是从左到右递增,从上往下递增。
        // 从左下角出发,target比当前小,上移动,target比当前大,右移动
      	// 防空arr[0]
      	if(matrix.length == 0) return false;
        int i = arr.length - 1;
        int j = 0;
        while (i >= 0 && j <= arr[0].length - 1) {
            if(target < arr[i][j]) {
                i--;
            }else if(target>arr[i][j]) {
                j++;
            }else {
                return true;
            }
        }
        return false;
    }
}

```



#### 29.顺时针打印矩阵

```java
package offer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
 */
public class offer_29_spiralOrder_rotate_顺时针打印矩阵 {
    public List<Integer> spiralOrder(int[][] matrix) {
        //定义行走的方向
        int[] rowBais = {0, 1, 0, -1};
        int[] colBais = {1, 0, -1, 0};
        int m = matrix.length;
        int n = matrix[0].length;
        int row = 0;
        int col = 0;
        //当前方向
        int dir = 0;
        //结果集
        List<Integer> res = new ArrayList<Integer>();
        //记录访问过的元素
        boolean[][] hasVisted = new boolean[m][n];
        for (int i = 0; i < m * n; i++) {
            res.add(matrix[row][col]);
            hasVisted[row][col] = true;
            //设想的下一步,判断是否需要移动方向
            int newRow = row + rowBais[dir];
            int newCol = col + colBais[dir];
            if (newRow < 0 || newRow >= m || newCol < 0 || newCol >= n || hasVisted[newRow][newCol]) {
                dir = (dir + 1) % 4;
            }
            row = row + rowBais[dir];
            col = col + colBais[dir];
        }
        return res;
    }

    public static void main(String[] args) {
        int[][] matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        List<Integer> result = new offer_29_spiralOrder_rotate_顺时针打印矩阵().spiralOrder(matrix);
        System.out.println(Arrays.toString(result.toArray()));
    }
}

```



#### 矩阵旋转90度

<img src="/Users/chenjing/Library/Application Support/typora-user-images/image-20220410213838718.png" alt="image-20220410213838718" style="zoom:33%;" />

```java
package offer;

/**
 * 正方形旋转90度
 * 每次交换4个元素
 * 如果是偶数的话一共要交换n^2个元素 一共n^2/4 = (n/2) * (n/2)
 * 如果是奇数的话中间一个元素不用交换,一共要交换n^2-1/4 = (n+1/2) * (n-1/2)
 *
 * (row,col) -> (col,n-row-1)
 * (col,n-row-1) -> (n-row-1,n-col-1)
 * (n-row-1,n-col-1) -> (n-col-1,row)
 * (n-col-1,row) -> (row,col)
 * @param
 */
public class offer_29_2_rotate_矩阵旋转90度 {

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //奇数的时候 n/2 = n-1/2
        //偶数的时候 n+1/2 = n/2
        for(int row = 0;row<n/2;row++) {
            for(int col = 0;col<(n+1)/2;col++) {
                int temp = matrix[row][col];
                matrix[row][col] = matrix[n-col-1][row];
                matrix[n-col-1][row] = matrix[n-row-1][n-col-1];
                matrix[n-row-1][n-col-1] = matrix[col][n-row-1];
                matrix[col][n-row-1] = temp;
            }
        }
    }

}

```







#### 50.第一个只出现一次的字符

```java
package offer;

/**
 * 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
 */
public class offer_50_firstNotRepeating_第一个只出现一次的字符 {
    //返回索引
    public int FirstNotRepeatingChar(String str) {
        int[] cnts = new int[128];
        for (int i = 0; i < str.length(); i++)
            cnts[str.charAt(i)]++;
        for (int i = 0; i < str.length(); i++)
            if (cnts[str.charAt(i)] == 1)
                return i;
        return -1;
    }

    //返回字符
    public char firstUniqChar(String s) {
        int[] dict = new int[128];
        for(int i = 0;i<s.length();i++) {
            dict[s.charAt(i)]++;
        }
        for(int i = 0;i<s.length();i++) {
            if(dict[s.charAt(i)] == 1){
                return s.charAt(i);
            }
        }
        return ' ';
    }

}
```



#### 5.替换空格

```java
package offer;

/**
 * 剑指 Offer 05. 替换空格
 * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
 * https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/
 * 有相关题目
 */
public class offer_05_replaceSpace_替换空格 {
    public static void main(String[] args) {
        String s = "We are happy.";
        String res = new offer_05_replaceSpace_替换空格().replaceSpace(s);
        System.out.println(res);
    }

    // 时间复杂度O(N)
    // 空间复杂度O(N) c++能到O(1)
    public String replaceSpace(String s) {
        StringBuilder sb = new StringBuilder();
        for(char c:s.toCharArray()) {
            if(c == ' ') {
                sb.append("%20");
            }else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * java的string不能resize
     * 先遍历统计空格的数量,扩容
     * 再倒序遍历,防止覆盖元素
     *
     *         int count = 0, len = s.size();
     *         // 统计空格数量
     *         for (char c : s) {
     *             if (c == ' ') count++;
     *         }
     *         // 修改 s 长度
     *         s.resize(len + 2 * count);
     *         // 倒序遍历修改
     *         for(int i = len - 1, j = s.size() - 1; i < j; i--, j--) {
     *             if (s[i] != ' ')
     *                 s[j] = s[i];
     *             else {
     *                 s[j - 2] = '%';
     *                 s[j - 1] = '2';
     *                 s[j] = '0';
     *                 j -= 2;
     *             }
     *         }
     *         return s;
     */
}
```





#### 21.奇数放到偶数前面

```java
package offer;

import java.util.Arrays;

/**
 * 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
 * 扩展 -1 0 1 排序
 */
public class offer_21_exchange_奇数放到偶数前面 {

    //双指针
    //left指针前面都是奇数
    public int[] exchange(int[] nums) {
        int left = 0, right = 0;
        while (right < nums.length) {
            if (nums[right] % 2 != 0) {
                swap(nums, left, right);
                left++;
            }
            right++;

        }
        return nums;
    }

    //双指针
    //左边找第一个偶数,右边找第一个奇数
    public int[] exchangeV2(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            while (nums[left] % 2 != 0) left++;
            while (nums[right] % 2 == 0) right--;
            swap(nums, left, right);
        }
        return nums;
    }

    //扩展 -1 0 1 排序
    //三个指针 左右不动,中间i动,找-1和左边换,1和右边换,0不换
    public int[] exchangeV3(int[] nums) {
        int left = 0, right = nums.length - 1, i = 0;
        //i不能大于right less从左到右走的,所以=-1的时候i可以+
        while (i <= right) {
            if (nums[i] == -1) {
                swap(nums, left++, i++);
            } else if (nums[i] == 1) {
                swap(nums, right--, i);
            } else {
                i++;
            }

        }
        return nums;
    }


    public void swap(int[] arr, int left, int right) {
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
    }

    public static void main(String[] args) {
//        int[] arr = {1, 3,2,4, 5};
        int[] arr = {1, -1, 0, -1, 0};
        new offer_21_exchange_奇数放到偶数前面().exchangeV3(arr);
        System.out.println(Arrays.toString(arr));
    }
}

```



#### 39.超过数组长度一半的数字

```java
package offer;

/**
 * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
 * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
 */
public class offer_39_moreThanHalfNum_超过数组长度一半的数字 {

    // 时间复杂度 O(N)
    // 空间复杂度 O(1)
    public int majorityElement(int[] nums) {
        int candidate = nums[0],vote = 1;
        for(int i = 1;i<nums.length;i++) {
            vote += candidate == nums[i]?1:-1;
            if(vote == 0) {
                candidate = nums[i];
                vote = 1;
            }
        }
        int count = 0;
        //验证超过半数
        for(int num:nums) {
            if(num == candidate) count++;
        }
        return count > nums.length/2?candidate:-1;
    }
}
```



#### 45.把数组排成最小的数

```java
package offer;

import java.util.Arrays;
import java.util.Comparator;

/**
 * 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
 */
public class offer_45_minNumber_把数组排成最小的数 {

    public String minNumber(int[] nums) {
        //排序小的往前 排序规则字符串相加
        String[] strs = new String[nums.length];
        for(int i = 0;i<nums.length;i++) {
            strs[i] = String.valueOf(nums[i]);
        }
//        Arrays.sort(strs,(o1,o2) -> (o1+o2).compareTo(o2+o1));
        Arrays.sort(strs, new Comparator<String>() {
            @Override
            public int compare(String o1,String o2) {
                return (o1+o2).compareTo(o2+o1);
            }
        });
        StringBuilder sb = new StringBuilder();
        for(int i = 0;i<strs.length;i++){
            sb.append(strs[i]);
        }
        return sb.toString();
    }
}
```



#### 51.逆序对

```java
package offer;

/**
 * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
 * 输入一个数组，求出这个数组中的逆序对的总数。
 * <p>
 * 1.冒泡排序调换顺序的时候就是个逆序对
 * 2.归并排序更好
 */
public class offer_51_reversePairs_逆序对 {

    private int count;
    private int[] tmp;

    public int reversePairs(int[] nums) {
        count = 0;
        tmp = new int[nums.length];
        mergeSort(nums, 0, nums.length-1);
        return count;
    }

    public void mergeSort(int[] nums, int lo, int hi) {
        if (lo >= hi) return;
        int mid = (lo + hi) / 2;
        mergeSort(nums, lo, mid);
        mergeSort(nums, mid + 1, hi);
        //合并
        merge(nums, lo, mid, hi);
    }

    public void merge(int[] nums, int lo, int mid, int hi) {
        int tmpIndex = lo;
        int i = lo;
        int j = mid + 1;
        while (i <= mid && j <= hi) {
            if (nums[i] <= nums[j]) {
                tmp[tmpIndex++] = nums[i++];
            } else {
                // 4 5 7    1 2 3
                count += (mid - i + 1);
                tmp[tmpIndex++] = nums[j++];
            }
        }
        while (i <= mid) {
            tmp[tmpIndex++] = nums[i++];
        }
        while (j <= hi) {
            tmp[tmpIndex++] = nums[j++];
        }

        for (tmpIndex = lo; tmpIndex <= hi; tmpIndex++) {
            nums[tmpIndex] = tmp[tmpIndex];
        }
    }
}
```



#### 53.1.数字在排序数组中出现的次数

```java
package offer;

/**
 * 统计一个数字在排序数组中出现的次数。
 */
public class offer_53_1_search_数字在排序数组中出现的次数 {

    public int search(int[] nums, int target) {
        int first = binarySearch(nums,target,true);
        if(first == -1) return 0;
        int last = binarySearch(nums,target,false);
        return last -first + 1;
    }

    public int binarySearch(int[] nums,int target,boolean first) {
        if (nums.length == 0 || nums == null) return -1;
        int left = 0;
        int right = nums.length - 1;
        int index = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                index = mid;;
                if(first) right = mid-1;
                else left = mid + 1;
            } else if ( target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return index;
    }

    public static void main(String[] args) {
        new offer_53_1_search_数字在排序数组中出现的次数().search(new int[]{1,2,3,4,5,6,6,7},6);
    }

}

```



#### 57.递增数组二数之和

```java
package offer;

/**
 * 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。
 * 如果有多对数字的和等于s，则输出任意一对即可。
 */
public class offer_57_sortTwoSum_递增数组二数之和 {

    //递增,用左右指针,相加结果大于target就右指针--,小于就左指针++
    public int[] twoSum(int[] nums,int target) {
        if(nums == null || nums.length <= 1) return null;
        //递增数组,用双指针
        int left = 0;
        int right = nums.length - 1;
        while(left < right) {
            if(nums[left] + nums[right] == target) {
                return new int[]{nums[left],nums[right]};
            }else if(nums[left] + nums[right] < target) {
                left++;
            }else {
                right--;
            }
        }
        return null;
    }
}

```



#### 58.1.翻转字符串

```java
package offer;

/**
 * 翻转字符串
 */
public class offer_58_1_reverseWords_翻转字符串 {

    public String reverseWords(String s) {
        String[] strs = s.trim().split(" ");
        StringBuilder res = new StringBuilder();
        for(int i = strs.length-1;i>=0;i--) {
            //遇到空单词跳过 "i am  a student"
            if(strs[i] == " ") continue;
            res.append(strs[i] + " ");
        }
        return res.toString().trim();
    }

    public static void main(String[] args) {
        String the_sky_is_blue = new offer_58_1_reverseWords_翻转字符串().reverseWords("i am a student.");
        System.out.println(the_sky_is_blue);
    }
}

```



#### 58.2.旋转数组_字符串

```java
package offer;

/**
 * 左旋转字符 左移
 	 输入: s = "abcdefg", k = 2
	 输出: "cdefgab"
 	
 * 右旋转字符 右移
 */
public class offer_58_2_rotateString_旋转数组_字符串 {
    public String reverseLeftWords(String s, int k) {
        if (s == null || s.length() == 0) return "";
        //防止k > s.length();
        k = k % s.length();
        char[] strs = s.toCharArray();
        reverse(strs, 0, strs.length - 1);
        reverse(strs, 0, strs.length - 1 - k);
        reverse(strs, strs.length - k, strs.length - 1);
        return String.valueOf(strs);
    }

    public String reverseRightWords(String s, int k) {
        if (s == null || s.length() == 0) return "";
        //防止k > s.length();
        k = k % s.length();
        char[] strs = s.toCharArray();
        reverse(strs, 0, strs.length - 1);
        reverse(strs, 0, k - 1);
        reverse(strs, k, strs.length - 1);
        return String.valueOf(strs);
    }

    public void reverse(char[] chars, int start, int end) {
        while (start < end) {
            char temp = chars[start];
            chars[start] = chars[end];
            chars[end] = temp;
            start++;
            end--;
        }
    }
}

```



#### 11.旋转数组的最小数字

```java
package offer;

/**
 * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
 * 给你一个可能存在重复元素值的数组numbers，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。
 * 请返回旋转数组的最小元素。例如，数组[3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为1。
 * 链接：https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof
 */
public class offer_11_rotataionMin_旋转数组的最小数字 {
    // 升序 旋转后,左边的大 右边的小
    // 如果没有重复元素
    public int rotataionMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }

    //时间复杂度 O(logn)
    //空间复杂度 O(1)
    public int rotationMin_repeat(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] > nums[right]) {
                left = mid + 1;
            }else if(nums[mid] < nums[right]) {
                right = mid;
            }else {
                right--;
            }
        }
        return nums[left];
    }
}

```



#### 61.扑克牌中的顺子

```java
package offer;

import java.util.Arrays;

/**
 * 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。
 * 2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
 * 链接：https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof
 */
public class offer_61_isStraight_扑克牌中的顺子 {

    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int cnt = 0;

        //统计0
        for(int i = 0;i<nums.length-1;i++) {
            if(nums[i] == 0) cnt++;
        }

        //i从非0开始
        //抵消0
        for(int i = cnt;i<nums.length-1;i++) {
            if(nums[i+1] == nums[i]) return false;
            cnt -= nums[i+1] - nums[i] - 1;
        }
        return cnt >= 0;
    }
}

```



#### 66.构造乘积数组

![image-20220414150232321](/Users/chenjing/Library/Application Support/typora-user-images/image-20220414150232321.png)

````java
package offer;

/**
 * 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，
 * 其中B[i] 的值是数组 A 中除了下标 i 以外的元素的积,
 * 即B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
 *
 * 链接：https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof
 */
public class offer_66_constructArr_构造乘积数组 {
    public int[] constructArr(int[] a) {
        int len = a.length;
        if(len == 0) return new int[0];
        int[] b = new int[len];
        b[0] = 1;

        //从1开始
        for(int i = 1; i < len; i++) {
            b[i] = b[i - 1] * a[i - 1];
        }
        int tmp = 1;
        //从len-2开始
        for(int i = len - 2; i >= 0; i--) {
            tmp *= a[i + 1];
            b[i] *= tmp;
        }
        return b;
    }

    public static void main(String[] args) {
        int[] arr = {1,2,3,4,5};
        int[] result = new offer_66_constructArr_构造乘积数组().constructArr(arr);
        System.out.println(result);
    }
}

````



#### 67.字符转数字

```java
package offer;

/**
 * 将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。
 * 数值为 0 或者字符串不是一个合法的数值则返回 0
 *
 * 注意：
 * ①字符串中可能出现任意符号，出现除 +/- 以外符号时直接输出 0
 * ②字符串中可能出现 +/- 且仅可能出现在字符串首位。
 *
 * https://www.nowcoder.com/practice/1277c681251b4372bdef344468e4f26e?tpId=13&tqId=11202&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking&from=cyc_github
 *
 * c- '0' 变数字
 * 5 + '0' 变字符
 */
public class offer_67_strToInt_字符转数字 {
    public int StrToInt(String str) {
        if(str == null||str.length() == 0) return 0;

        boolean isNegtive = str.charAt(0) == '-';
        int res = 0;
        for(int i= 1;i<str.length();i++) {
            char c = str.charAt(i);
            if(i == 0&&(c == '+' || c=='-')) continue;
            if(c < '0' || c>'9') return 0;
            res = res * 10 + (c - '0');
        }
        return isNegtive?-res:res;
    }
}

```



#### 16.数值的整数次方

```java
package offer;

/**
 * 实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn ）。
 * 示例 1：
 * 输入：x = 2.00000, n = 10
 * 输出：1024.00000
 * https://leetcode-cn.com/problems/powx-n/
 */
public class offer_16_myPow_数值的整数次方 {
    public double myPow(double x, int n) {
        long N = n;
        return pow(x,N);
    }

    // 时间复杂度 O(n)
    // 空间复杂度 O(1)
    // 分为负数、正偶数、正奇数
    public double pow(double x,long n ) {
        if(n == 0) return 1;
        if(n == 1) return x;
        //转为正数
        if(n < 0) return pow(1/x,-n);
        if(n % 2 == 0) {
            return pow(x*x,n/2);
        }else {
            return pow(x,n-1)*x;
        }
    }

}

```



#### 17.打印1到最大的n位数

```java
package offer;

/**
 * 打印从1到最大的n位数 n可能是大数
 * (char)0 和 (char)0+'0'不一样
 *
 * https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/
 *
 * 时间复杂度 O(10^n)
 * 空间复杂度 O(10^n)
 */
public class offer_17_print1ToMaxOfNDigits_打印1到最大的n位数 {
    public void print1ToMaxOfNDigits(int n) {
        if (n <= 0) {
            return;
        }
        char[] number = new char[n];
        print1ToMaxOfNDigits(number, 0);
    }

    private void print1ToMaxOfNDigits(char[] number, int digit) {
        if (digit == number.length) {
            // 第一次00 第二次01 第三次02
            printNumber(number);
            return;
        }
        for (int i = 0; i < 10; i++) {
            number[digit] = (char) (i + '0');
            print1ToMaxOfNDigits(number, digit + 1);
        }
    }

    private void printNumber(char[] number) {
        boolean isBegining = true;
        for (int i = 0; i < number.length; i++) {
            if(isBegining&&number[i] != '0') {
                isBegining = false;
            }
            if(!isBegining) {
                System.out.print(number[i]);
            }

        }
        System.out.println();
    }

    public static void main(String[] args) {
        new offer_17_print1ToMaxOfNDigits_打印1到最大的n位数().print1ToMaxOfNDigits(2);
    }
}

```



#### 20.字符串是否是数值

```java
package offer;

/**
 * 剑指 Offer 20. 表示数值的字符串
 * 123.
 * .1
 * 123.45e+6
 *
 *
 */
public class offer_20_isNumber_字符串是否是数值 {

    public boolean isNumber(String s) {
        // 1.先去空格
        s = s.trim();
        // 2.判断是否是空
        if (s.length() == 0) return false;
        // 3.去掉 + -
        if (s.charAt(0) == '+' || s.charAt(0) == '-') {
            s = s.substring(1);
        }
        // 4.分割e
        // e左边可能是小数,e右边是整数且可能有+-号
        s = s.replace('E', 'e');
        if (s.indexOf('e') >= 0) {
            int index = s.indexOf('e');
            String left = s.substring(0, index);
            String right = s.substring(index + 1);
            // e右边+-号
            if (right.length() > 0) {
                if (right.charAt(0) == '+' || right.charAt(0) == '-') {
                    right = right.substring(1);
                }
            }
            return validNumber(left) && validPureNumber(right);
        }
        return validNumber(s);
    }

    //验证整数
    public boolean validPureNumber(String s) {
        if(s.length() == 0) return false;
        char[] chars = s.toCharArray();
        for(char c: chars) {
            if(c < '0' || c > '9') return false;
        }
        return true;
    }

    //验证小数
    public boolean validNumber(String s) {
        if(s.indexOf('.') >= 0) {
            int index = s.indexOf('.');
            String left = s.substring(0,index);
            String right = s.substring(index+1);
            //防止左右为空 .1 1.
            if(left.length() > 0 && right.length() > 0) {
                return validPureNumber(left) && validPureNumber(right);
            }else if(left.length() > 0) {
                return validPureNumber(left);
            }else if(right.length() > 0) {
                return validPureNumber(right);
            }else {
                return false;
            }
        }
        return validPureNumber(s);
    }


    public static void main(String[] args) {
        boolean result = new offer_20_isNumber_字符串是否是数值().isNumber(".1");
        System.out.println(result);
    }
}

```





#### 62. 圆圈中最后剩下的数

```java
//约瑟夫环，圆圈长度为 n 的解可以看成长度为 n-1 的解再加上报数的长度 m。因为是圆圈，所以最后需要对 n 取余。
public int LastRemaining_Solution(int n, int m) {
    if (n == 0)     /* 特殊输入的处理 */
        return -1;
    if (n == 1)     /* 递归返回条件 */
        return 0;
    return (LastRemaining_Solution(n - 1, m) + m) % n;
}
```



### 7.动态规划

> 解题步骤

1. 确定dp数组以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序

> 适合题目

1. 计数(有多少种方式/方法)
2. 求最大值和最小值
3. 求存在性

#### 2.1 斐波那契数列

```
写一个函数，输入n，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：
F(0) = 0, F(1)= 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

```java
public class Fabnacci {
  	//时间复杂度O(n)
  	//空间复杂度O(n)
    public int fib(int n) {
        //1.确定dp数组和下标的含义 dp[i]:第i的斐波那契数组的值
        //2.确定递推公式 题干已经给出F(N) = F(N - 1) + F(N - 2)
        //3.dp数组如何初始化 题干已经给出 F(0) = 0, F(1)= 1
        //4.确定遍历顺序 从前到后遍历的
        if (n <= 1) return n;
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2]; //要算n项,new dp[n+1]
        }
        return dp[n];
    }
  
		//时间复杂度O(n)
  	//状态压缩空间复杂度O(1)
    public int fib_V2(int n) {
        if (n <= 1) return n;
        int a = 0, b = 1, c = 0;
        for (int i = 2; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        return c;
    }

    public static void main(String[] args) {
        int result = new Fabnacci().fib_V2(3);
        System.out.println(result);
    }
}
```



#### 2.2 爬楼梯

```
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
```

```
分析递推公式：
	1层 1种
	2层 2种
	3层 因为每次可以爬1或2层,所以只能从1层和2层爬上来,所以是1层(1种)+2层(2种) => 当前层和前2层有关系
初始化：
	dp[0] = 0;
	dp[1] = 1;
```



```java
public class ClimbStairs {
    public int climbStairs(int n) {
        //1.dp数组含义 dp[i]爬到i层有多少种方法
        //2.确定递推公式 dp[i] = dp[i-1]+dp[i-2]
        //3.dp初始化 dp[1]=1 dp[2] = 2
        //4.遍历顺序
        if (n <= 2) return n;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int climbStairs_V2(int n) {
        if (n <= 2) return n;
        int a = 1, b = 2, c = 0;
        for (int i = 3; i <= n; i++) {
            c = a + b;
            a = b;
            b = c;
        }
        return c;
    }

    public static void main(String[] args) {
        int result = new ClimbStairs().climbStairs(3);
        System.out.println(result);
    }
```



#### 2.3 背包问题



![package](../pic/package.png)



```java
/**
 * 01 背包问题
 */
public class package01 {
    public static void main(String[] args) {
        int[] weight = {1, 3, 4};
        int[] value = {15, 20, 30};
        int bagsize = 4;
        testweightbagproblem(weight, value, bagsize);
    }

//    public static void testweightbagproblem(int[] weight, int[] value, int bagsize) {
//        int wlen = weight.length;
//        //定义dp数组：dp[i][j]表示背包容量为j时，前i个物品能获得的最大价值
//        //i代表物品 j代表价值(多个0)
//        int[][] dp = new int[wlen][bagsize + 1];
//        //初始化物品为0时候的情况
//        for (int j = 0; j <= bagsize; j++) {
//            dp[0][j] = j >= weight[0] ? value[0] : 0;
//        }
//        //遍历顺序：先遍历物品，再遍历背包容量
//        for (int i = 1; i < wlen; i++) {
//            for (int j = 0; j <= bagsize; j++) {
//                if (j < weight[i]) {
//                    dp[i][j] = dp[i - 1][j];
//                } else {
//                    dp[i][j] = Math.max(dp[i - 1][j], value[i] + dp[i - 1][j - weight[i]]);
//                }
//            }
//        }
//        //打印dp数组
//        for (int i = 0; i < wlen; i++) {
//            for (int j = 0; j <= bagsize; j++) {
//                System.out.print(dp[i][j] + " ");
//            }
//            System.out.print("\n");
//        }
//    }


//    public static void testweightbagproblem(int[] weight, int[] value, int bagsize) {
//        int wlen = weight.length;
//        int[] dp = new int[bagsize + 1];
//        //初始化物品为0时候的情况
//         for (int j = bagsize; j >= 0; j--) {
//            dp[j] = j >= weight[0] ? value[0] : 0;
//         }
//        //遍历顺序：先遍历物品，再遍历背包容量
//        for (int i = 1; i < wlen; i++) {
//            for (int j = bagsize; j >= 0; j--) {
//                if (j >= weight[i]) dp[j] = Math.max(dp[j], value[i] + dp[j - weight[i]]);
//            }
//        }
//        //打印dp
//        for (int j = 0; j <= bagsize; j++) {
//            System.out.print(dp[j] + " ");
//        }
//        System.out.print("\n");
//
//    }

    public static void testweightbagproblem(int[] weight, int[] value, int bagsize) {
        int wlen = weight.length;
        int[] dp = new int[bagsize + 1];

        //遍历顺序：先遍历物品，再遍历背包容量(初始化包含在里面)
        for (int i = 0; i < wlen; i++) {
            for (int j = bagsize; j >= weight[i]; j--) {
                dp[j] = Math.max(dp[j], value[i] + dp[j - weight[i]]);
            }
        }
        //打印dp
        for (int j = 0; j <= bagsize; j++) {
            System.out.print(dp[j] + " ");
        }
        System.out.print("\n");

    }
}
```



#### 2.4 打家劫舍

> 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
>
> 给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
>
> 示例 1：
>
> 输入：[1,2,3,1]
> 输出：4
> 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
> 偷窃到的最高金额 = 1 + 3 = 4 。



```java
class Solution {
    public int rob(int[] nums) {
        if(nums.length == 1) return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0],nums[1]);
        for(int j = 2;j<nums.length;j++) {
          	//不偷dp[j-1],偷dp[j-2]+nums[j]
            dp[j] = Math.max(dp[j-1],dp[j-2]+nums[j]);
        }
        return dp[nums.length-1];
    }
}
```



#### 10.跳台阶

```java
package offer;

/**
 * 青蛙跳台阶
 * 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n级的台阶总共有多少种跳法。
 * <p>
 * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
 * <p>
 * 链接：https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof
 */
public class offer_10_02_stairs_跳台阶 {


    public static void main(String[] args) {
        int res = new offer_10_02_stairs_跳台阶().stairs(5);
        System.out.println(res);
    }


    /**
     * 动态规划 优化
     * 时间复杂度 O(n)
     * 空间复杂度 O(1)
     *
     * @param n
     * @return
     */
    public int stairs(int n) {
        // 0阶算1种方法
        int a = 1, b = 1, sum;
        for(int i = 2; i < n; i++){
            sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }

    // 常规方式
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
  	
  	/**
  	*跳上 n-1 级台阶，可以从 n-2 级跳 1 级上去，也可以从 n-3 级跳 2 级上去...，那么

		f(n-1) = f(n-2) + f(n-3) + ... + f(0)
		同样，跳上 n 级台阶，可以从 n-1 级跳 1 级上去，也可以从 n-2 级跳 2 级上去... ，那么

		f(n) = f(n-1) + f(n-2) + ... + f(0)
		综上可得

		f(n) - f(n-1) = f(n-1)
		即f(n) = 2*f(n-1)
		所以 f(n) 是一个等比数列

		public int JumpFloorII(int target) {
    	return (int) Math.pow(2, target - 1);
		}
		/
    //一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级... 它也可以跳上 n 级。
    //求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
  	public int jumpFloorII(int target) {
    	int[] dp = new int[target];
    	Arrays.fill(dp, 1);
    	for (int i = 1; i < target; i++)
        	for (int j = 0; j < i; j++)
            	dp[i] += dp[j];
    	return dp[target - 1];
	  }

}

```



#### 42. 连续子数组的最大和

```java
package offer;

/**
 * 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
 *
 * 要求时间复杂度为O(n)。
 */
public class offer_42_maxSubArray_连续子数组的最大和_动态规划 {
    public int maxSubArray(int[] nums) {
        //dp[i] 包含i在内的子数组 和 最大值为dp[i]   并不是说一定以下标0为起始位置。
        int[] dp = new int[nums.length];
        //初始化 连续数组必须包含当前元素
        dp[0] = nums[0];
        int maxSum = nums[0];
        //递推公式 dp[i] = Math.max(dp[i-1]+nums[i],nums[i])
        // maxsum = Math.max(dp[i],maxSum)
        for(int i = 1;i<nums.length;i++) {
            dp[i] = Math.max(dp[i-1] + nums[i],nums[i]);
            maxSum = Math.max(dp[i],maxSum);
        }
        return maxSum;
    }
}
```



#### 连续自增子数组

![image-20220413163736119](../pic/连续递增子数组.png)

```java
class Solution {
    public int findLengthOfLCIS(int[] nums) {
        //dp[i] 以i结尾的子数组长度
        int[] dp = new int[nums.length];
        //初始化
        dp[0] = 1;
        int maxLen = dp[0];
        for(int i = 1;i<nums.length;i++) {
            if(nums[i] > nums[i-1]) {
                dp[i] = dp[i-1] + 1;
            }else {
                dp[i] = 1;
            }
            maxLen = Math.max(dp[i],maxLen);
        }
        return maxLen;

    }
}
```



#### 连续递增子序列

![image-20220413171032614](../pic/连续递增子序列.png)

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        //dp[i] 以i结尾的递增序列最大长度
        int[] dp = new int[nums.length];
        Arrays.fill(dp,1);
        //初始化
        dp[0] = 1;
        int maxLen = dp[0];
        for(int j = 1;j<nums.length;j++) {
            for(int i = 0;i<j;i++) {
                if(nums[j] > nums[i]) {
                    dp[j] = Math.max(dp[i]+1,dp[j]);
                    maxLen = Math.max(dp[j],maxLen);
                }
            }
        }
        return maxLen;
    }
}
```



#### 47. 礼物的最大价值

```java
package offer;

/**
 * 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。
 * 你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。
 * 给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
 * <p>
 * 链接：https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof
 */
public class offer_47_maxValue_礼物的最大值 {
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        //dp[i][j] 到ij位置最大价值
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        //初始化
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] += dp[0][j - 1] + grid[0][j];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(dp[i][j] + " ");
            }
            System.out.println();
        }
        return dp[m - 1][n - 1];
    }

    public static void main(String[] args) {
        int[][] arr = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
        int result = new offer_47_maxValue_礼物的最大值().maxValue(arr);
    }
}
```



#### 48. 最长不含重复字符的子字符串

```java
package offer;

import java.util.Arrays;

/**
 * 最长不含重复字符的子字符串
 */
public class offer_48_lengthOfLongestSubstring_最长不含重复字符的子字符串 {

    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        if(n <= 1) return n;
        int left = 0,right = 0;
        //ascii
        int[] window = new int[128];
        Arrays.fill(window,-1);
        int maxLen = 1;
        while(right < n) {
            char rightChar = s.charAt(right);
            //没有就是-1,有就是有
            int rightIndex = window[rightChar];
            //如果包含就移动left  left移动到重复的下个位置 left不能移动到left之前
            left = Math.max(left,rightIndex+1);
            maxLen = Math.max(maxLen,right-left+1);
            window[rightChar] = right;
            right++;
        }
        return maxLen;
    }

    public static void main(String[] args) {
        String s = "au";
        new offer_48_lengthOfLongestSubstring_最长不含重复字符的子字符串().lengthOfLongestSubstring(s);
    }
}
```



#### 49. 丑数

```java
package offer;

/**
 * 把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。
 * 例如 6、8 都是丑数，但 14 不是，因为它包含因子 7。习惯上我们把 1 当做是第一个丑数。
 * 求按从小到大的顺序的第 N 个丑数。
 */
public class offer_49_ugly_丑数 {

    /**
     * 判断是否是丑数
     *
     * @param num
     * @return
     */
    public boolean isUgly(int num) {
        if (num <= 0) return false;
        while (num % 2 == 0) num /= 2;
        while (num % 3 == 0) num /= 3;
        while (num % 5 == 0) num /= 5;
        return num == 1;
    }

    public int ugly(int num) {
        int i2 = 0, i3 = 0, i5 = 0;
        // dp[i] dp[i]代表第 i+1 个丑数；
        int[] dp = new int[num];
        dp[0] = 1;
        for (int i = 1; i < num; i++) {
            dp[i] = Math.min(Math.min(dp[i2] * 2, dp[i3] * 3), dp[i5] * 5);
            if(dp[i] == dp[i2]*2) i2++;
            if(dp[i] == dp[i3]*3) i3++;
            if(dp[i] == dp[i5]*5) i5++;
        }
        return dp[num-1];
    }
}
```



#### 63.股票的最大利润

```java
package offer;

public class offer_63_maxProfit_股票的最大利润 {
    public int maxProfit(int[] prices) {
        if(prices.length == 0) return 0;
        //dp[i][0] 第i天结束时候,没持股的最大利润
        //dp[i][1] 第i天结束时候,持股的最大利润
        int[][] dp = new int[prices.length][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        //dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1] + prices[i])
        //dp[i][1] = Math.max(dp[i-1][1],dp[i-1][0] - prices[i])
        //因为只能交易一次所以 dp[i-1][0] = 0
        //dp[i][1] = Math.max(dp[i-1][1],-prices[i])

        //状态压缩 0和 01有关,1只和1有关
        //dp[0] = Math.max(dp[0],dp[1]+prices[i]); a = Math.max(a,b+price[i]);
        //dp[1] = Math.max(-prices[i],dp[1]);      b = Math.max(-prices[i],b);
        for(int i = 1;i<prices.length;i++) {
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1] + prices[i]);
            dp[i][1] = Math.max(-prices[i],dp[i-1][1]);
        }
        return dp[prices.length-1][0];
    }

    public int maxProfit_V2(int[] prices) {
        if(prices.length==0) return 0;
        // a没持股,b持股
        int a = 0,b = -prices[0];
        for(int i = 1;i<prices.length;i++) {
            a = Math.max(a,b+prices[i]);
            b = Math.max(-prices[i],b);
        }
        return a;
    }
}

```



#### 46.把数字翻译成字符串

```java
package offer;
/*
 * 有一种将字母编码成数字的方式：'a'->1, 'b->2', ... , 'z->26'。
 * 我们把一个字符串编码成一串数字，再考虑逆向编译成字符串。
 * 由于没有分隔符，数字编码成字母可能有多种编译结果，例如 11 既可以看做是两个 'a' 也可以看做是一个 'k' 。
 * 但 10 只可能是 'j' ，因为 0 不能编译成任何结果。
 * 现在给一串数字，返回有多少种可能的译码结果
*/
public class offer_46_translateNum_把数字翻译成字符串 {
    public int translateNum(int num) {
        String s = String.valueOf(num);
        //12258 dp[0]不存在,所以第一位是dp[1]
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= s.length(); i ++){
            String temp = s.substring(i-2, i);
            if(temp.compareTo("10") >= 0 && temp.compareTo("25") <= 0)
                dp[i] = dp[i-1] + dp[i-2];
            else
                dp[i] = dp[i-1];
        }
        return dp[s.length()];
    }

    public static void main(String[] args) {
        new offer_46_translateNum_把数字翻译成字符串().translateNum(12258);
    }
}

```



### bfs

#### 岛屿问题

```java
package bfs;

import java.util.LinkedList;
import java.util.Queue;

/**
 * 200. 岛屿数量
 * 给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
 * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
 * 此外，你可以假设该网格的四条边均被水包围。
 * 链接：https://leetcode-cn.com/problems/number-of-islands
 * <p>
 * 技巧1：xBias,yBias 棋盘问题
 * 技巧2：把二维坐标压缩成1个数 x = i*n+j
 * i = x/n
 * j = x%n
 * 技巧3：如果需要把每一层的结果放到一起,或者统计经过了几层,就需要用queue.size()来遍历
 *
 * 注：需要多条路径同时可以走的题目用bfs,剑指 Offer 12. 矩阵中的路径不满足
 */
public class lc_200_numIslands {
    int[] xBias = {0, 0, 1, -1};
    int[] yBias = {1, -1, 0, 0};

    public int numIsLands(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    // 目的：消灭相邻的1为0
                    bfs(grid, i, j);
                }
            }
        }
        return count;
    }

    public void bfs(char[][] grid, int i, int j) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(i * n + j);
        grid[i][j] = '0';

        while (!queue.isEmpty()) {
            Integer curr = queue.poll();
            i = curr / n;
            j = curr % n;
            for (int k = 0; k < xBias.length; k++) {
                int newI = i + yBias[k];
                int newJ = j + xBias[k];
                if (newI < 0 || newI >= m || newJ < 0 || newJ >= n || grid[newI][newJ] == '0') continue;
                queue.offer(newI * n + newJ);
                grid[newI][newJ] = '0';
            }
        }
    }

}

```

#### 课程表



```java
/**
 * 统计课程安排图中每个节点的入度，生成 入度表 indegrees。
 * 借助一个队列 queue，将所有入度为 00 的节点入队。
 * 当 queue 非空时，依次将队首节点出队，在课程安排图中删除此节点 pre：
 * 并不是真正从邻接表中删除此节点 pre，而是将此节点对应所有邻接节点 cur 的入度 -1−1，即 indegrees[cur] -= 1。
 * 当入度 -1−1后邻接节点 cur 的入度为 00，说明 cur 所有的前驱节点已经被 “删除”，此时将 cur 入队。
 * 在每次 pre 出队时，执行 numCourses--；
 * 若整个课程安排图是有向无环图（即可以安排），则所有节点一定都入队并出队过，即完成拓扑排序。
   换个角度说，若课程安排图中存在环，一定有		节点的入度始终不为 00。
 * 因此，拓扑排序出队次数等于课程个数，返回 numCourses == 0 判断课程是否可以成功安排。
 *
 */
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        List<List<Integer>> adjacency = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < numCourses; i++) {
          adjacency.add(new ArrayList<>());
        }
        // Get the indegree and adjacency of every course.
        for(int[] cp : prerequisites) {
            indegrees[cp[0]]++;
            adjacency.get(cp[1]).add(cp[0]);
        }
        // Get all the courses with the indegree of 0.
        for(int i = 0; i < numCourses; i++){
          	if(indegrees[i] == 0) queue.add(i);
        }
        // BFS TopSort.
        while(!queue.isEmpty()) {
            int pre = queue.poll();
            numCourses--;
            for(int cur : adjacency.get(pre))
                if(--indegrees[cur] == 0) queue.add(cur);
        }
        return numCourses == 0;
    }
}
```



### 排序

#### 冒泡排序

```java
package sort.unrecommend;

/**
 * 时间复杂度O(N2) 最好O(N)
 * 空间复杂度O(1)
 * 稳定排序算法
 */
public class BubbleSort {
    public static void main(String[] args) {
        int[] arr = {3, 6, 2, 8, 5};
        new BubbleSort().bubbleSort_V2(arr);
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i]);
        }
    }

    /**
     * 普通冒泡排序
     *
     * @param arr
     */
    public void bubbleSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {          //冒泡排序的趟数,第一个就是最小的,不用比
            for (int j = 0; j < arr.length - 1 - i; j++) {  //无序区[0,length-1-i),每比一趟确定一个值的最终位置,后面少一个需要比的
                if (arr[j] > arr[j + 1]) {
                    int tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                }
            }
        }
    }

    /**
     * 提前结束排序的优化
     *
     * @param arr
     */
    public void bubbleSort_V1(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            //是否发生交换标识
            boolean flag = false;
            for (int j = 0; j < arr.length - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    int tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                    flag = true;
                }
            }
            //没有发生交换标识提前结束排序
            if (!flag) {
                break;
            }
        }
    }

    /**
     * 优化点2 有序区边界
     *
     * @param arr
     */
    public void bubbleSort_V2(int[] arr) {
        // 发生交换的位置
        int lastChangeIndex = 0;
        // 有序区的边界
        int sortBorder = arr.length - 1;
        for (int i = 0; i < arr.length - 1; i++) {
            //是否发生交换标识
            boolean flag = false;
            for (int j = 0; j < sortBorder; j++) {
                if (arr[j] > arr[j + 1]) {
                    int tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                    flag = true;
                    lastChangeIndex = j;
                }
            }
            //没有发生交换标识提前结束排序
            if (!flag) {
                break;
            }
            //更新边界
            sortBorder = lastChangeIndex;
        }
    }

}
```



#### 选择排序

```java
package sort.unrecommend;

/**
 * 选择排序
 * 时间O(N2)
 * 空间O(1)
 * 不稳定
 */
public class SelectSort {
    public int[] sortArray(int[] nums) {
        if (nums == null || nums.length < 2) return nums;
        for (int i = 0; i < nums.length - 1; i++) {   //轮数,最后一个不用比,就是最大的
            int minNumIndex = i;
            for (int j = i + 1; j < nums.length; j++) { //无序区[i+1,nums.length)找到最小值,和i替换,每比一轮前面少一个要比的
                if (nums[j] < nums[minNumIndex]) {
                    minNumIndex = j;
                }
            }
            int tmp = nums[i];
            nums[i] = nums[minNumIndex];
            nums[minNumIndex] = tmp;
        }
        return nums;
    }
}
```



#### 插入排序

```java
package sort;

import java.util.Arrays;

/**
 * 插入排序
 */
public class InsertSort {
    public int[] sortArray(int[] nums) {
        if (nums == null || nums.length < 2) return nums;
        for (int i = 1; i < nums.length; i++) {
            int tmp = nums[i];
            int j;
            for (j = i; j > 0; j--) {
                if (tmp < nums[j - 1]) {
                    nums[j] = nums[j - 1];
                } else {
                    break;
                }
            }
            nums[j] = tmp;
        }
        return nums;
    }

    public static void main(String[] args) {
        int[] arr = {3, 6, 2, 8, 5};
        int[] result = new InsertSort().sortArray(arr);
        System.out.println(Arrays.toString(result));
    }
}

```



#### 快排

```java
package sort;
import java.util.*;
/**
 * Arrays.sort()
 * 1.基本数据类型：
 *  1小于47 插入排序(稳定)
 *  47到286 递归的快速排序(不稳定,基本数据类型不在乎稳定,例如3和3是一样的)
 *  大于286
 *      检查是否具备结构(时增时减超过67次),不具备结构还是用快速排序
 *      迭代(自底朝上)实现的归并排序
 * 2.引用数据类型：(1.类实现comparable,重写compareTo方法. 2.自定义comparator比较器)
 *  小数据量 插入排序
 *  大数据量 迭代(自底朝上)实现的归并排序
 *
 *
 *  O(n2) 冒泡(移动操作太多) < 选择(不稳定)  < 插入(推荐)
 *  O(n3/2) 希尔排序(不稳定)
 *  O(nlogn)
 *        归并        空间O(n) 稳定的排序算法
 *        快排(不稳定) 空间O(logn) 但时间复杂度最坏O(n2),且不稳定
 *        堆排        空间O(1) 对cache不友好,不推荐
 *  O(n) 桶排序 计数排序 基数排序

 * log的底 普通应用都是10，计算机学科是2，编程语言里面是e。
 */
public class QuickSort {

    public static void main(String[] args) {
        int[] arr = {3, 1, 2, 5, 7, 4, 6};
        int[] nums = new QuickSort().sortArray(arr);
        System.out.println(Arrays.toString(nums));
    }

    public int[] sortArray(int[] nums) {
        if (nums == null || nums.length < 2) return nums;
        quickSort(nums, 0, nums.length - 1);
        return nums;
    }

    public void quickSort(int[] nums, int lo, int hi) {
        if (lo >= hi) return;
        //计算pos
        int pos = partition(nums, lo, hi);
        quickSort(nums, lo, pos - 1);
        quickSort(nums, pos + 1, hi);
    }

    public int partition(int[] nums, int lo, int hi) {
        int rand = new Random().nextInt(hi - lo + 1) + lo;
        swap(nums, rand, hi);
        int pivot = nums[hi];
        int less = lo;
        int great = lo;
        for (; great <= hi - 1; great++) {
            if (nums[great] < pivot) {
                swap(nums, less, great);
                less++;
            }
        }
        swap(nums, less, hi);
        return less;
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}

```



#### 三路快排

```java
public int[] sortArray(int[] nums) {
  if(nums.length < 2) return nums;
  quickSort(nums,0,nums.length-1);
  return nums;
}

public void quickSort(int[] nums,int lo,int hi) {
  if(lo >= hi) return;
  int randIndex = new Random().nextInt(hi-lo+1) + lo;
  swap(nums,hi,randIndex);
  int pivot = nums[hi];
  int less = lo,great = hi,i = lo;
  while(i <= great) {
    if(nums[i] < pivot) {
      swap(nums,i,less);
      i++;
      less++;
    }else if(nums[i] > pivot) {
      swap(nums,i,great);
      great--;
    }else {
      i++;
    }
  }
  quickSort(nums,lo,less-1);
  quickSort(nums,great+1,hi);
}

public void swap(int[] nums,int i,int j) {
  int temp = nums[i];
  nums[i] = nums[j];
  nums[j] = temp;
}
```



#### 归并排序

```java
class Solution {
    private int[] temp;

    public int[] sortArray(int[] nums) {
        if(nums.length <= 1) return nums;
        temp = new int[nums.length];
        mergeSort(nums,0,nums.length-1);
        return nums;
    }

    public void mergeSort(int[] nums,int lo,int hi) {
        if(lo >= hi) return;
        int mid = (lo+hi)/2;
        mergeSort(nums,lo,mid);
        mergeSort(nums,mid+1,hi);
        merge(nums,lo,mid,hi);
    }

    public void merge(int[] nums,int lo,int mid,int hi) {
        int i = lo,j=mid+1,tempIndex = lo;
        while(i<=mid&&j<=hi) {
            if(nums[i] <= nums[j]) {
                temp[tempIndex++] = nums[i++];
            }else {
                temp[tempIndex++] = nums[j++];
            }
        }
        while(i<=mid) {
            temp[tempIndex++] = nums[i++]; 
        }
        while(j<=hi) {
            temp[tempIndex++] = nums[j++];
        }

        for(tempIndex=lo;tempIndex<=hi;tempIndex++) {
            nums[tempIndex] = temp[tempIndex];
        }
    }

    
}
```



#### 堆排

```java
/**
 * 堆排序
 */
public class HeapSort {

    public int[] sortArray(int[] nums) {
        int len = nums.length;
        // 将数组整理成堆
        heapify(nums);

        // 循环不变量：区间 [0, i] 堆有序
        for (int i = len - 1; i >= 1; ) {
            // 把堆顶元素（当前最大）交换到数组末尾
            swap(nums, 0, i);
            // 逐步减少堆有序的部分
            i--;
            // 下标 0 位置下沉操作，使得区间 [0, i] 堆有序
            siftDown(nums, 0, i);
        }
        return nums;
    }

    /**
     * 将数组整理成堆（堆有序）
     */
    private void heapify(int[] nums) {
        int len = nums.length;
        // 只需要从 i = (len - 1) / 2 这个位置开始逐层下移
        for (int i = (len - 1) / 2; i >= 0; i--) {
            siftDown(nums, i, len - 1);
        }
    }

    /**
     * @param nums
     * @param k    当前下沉元素的下标
     * @param end  [0, end] 是 nums 的有效部分
     */
    private void siftDown(int[] nums, int k, int end) {
        while (2 * k + 1 <= end) {
            int j = 2 * k + 1;
            if (j + 1 <= end && nums[j + 1] > nums[j]) {
                j++;
            }
            if (nums[j] > nums[k]) {
                swap(nums, j, k);
            } else {
                break;
            }
            k = j;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

}

```

