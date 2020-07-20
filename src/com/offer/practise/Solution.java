package com.offer.practise;

import javax.swing.*;
import java.util.*;

/**
 * ClassName:Solution
 * Package:com.offer.practise
 * Description:
 *
 * @Date:2020/2/29 下午 5:34
 * @Author:gaochenyu2020@163.com
 */
public class Solution {

    /**
     * 顺时针打印矩阵
     * @param matrix
     * @return
     */
    public ArrayList<Integer> printMatrix(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return null;
        }
        int rows = matrix.length;
        int columns = matrix[0].length;
        ArrayList<Integer> list = new ArrayList<>();
        int start = 0;
        while (columns > 2 * start && rows > 2 * start) {
            addMatrixInCircle(matrix, columns, rows, start, list);
            start++;
        }
        return list;
    }

    /**
     * 顺时针打印一圈
     * @param matrix
     * @param columns
     * @param rows
     * @param start
     * @param list
     */
    public void addMatrixInCircle(int[][] matrix, int columns, int rows, int start, ArrayList list) {
        int endX = columns - 1 - start;
        int endY = rows - 1 - start;

        for (int i = start; i <= endX; i++) {
            list.add(matrix[start][i]);
        }

        if (start < endY) {
            for (int i = start + 1; i <= endY; i++) {
                list.add(matrix[i][endX]);
            }
        }

        if (start < endX && start < endY) {
            for (int i = endX - 1; i >= start; i--) {
                list.add(matrix[endY][i]);
            }
        }

        if (start < endX && start < endY - 1) {
            for (int i = endY - 1; i > start; i--) {
                list.add(matrix[i][start]);
            }
        }
    }

    /**
     * 包含min函数的栈
     */
    Stack mData = new Stack();
    Stack mMin = new Stack();

    public void push(int node) {
        mData.push(node);
        if (mMin.isEmpty() || node < (int) mMin.peek()) {
            mMin.push(node);
        } else {
            mMin.push((int) mMin.peek());
        }
    }

    public void pop() {
        if (mData.size() > 0 && mMin.size() > 0) {
            mMin.pop();
            mData.pop();
        }
    }

    public int top() {
        return (int) mData.peek();
    }

    public int min() {
        return (int) mMin.peek();
    }

    /**
     * 栈的压入、弹出序列
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0 || popA.length != pushA.length) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        for (int i = 0, j = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            while (!stack.isEmpty() && stack.peek() == popA[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 从上往下打印二叉树
     */
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;
        }
    }

    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        ArrayList<TreeNode> queue = new ArrayList<>();
        if (root == null) {
            return list;
        }
        queue.add(root);
        while (queue.size() != 0) {
            TreeNode temp = queue.remove(0);
            if (temp.left != null) {
                queue.add(temp.left);
            }
            if (temp.right != null) {
                queue.add(temp.right);
            }
            list.add(temp.val);
        }
        return list;
    }

    /**
     * 二叉搜索树的后序遍历序列
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0) {
            return false;
        }
        if (sequence.length == 1) {
            return true;
        }
        return ju(sequence, 0, sequence.length - 1);

    }

    private boolean ju(int[] a, int start, int end) {
        if (start >= end) {
            return true;
        }
        int i = end;
        while (i > start && a[i - 1] > a[end]) {
            i--;
        }
        for (int j = start; j <= i - 1; j++) {
            if (a[j] > a[end]) {
                return false;
            }
        }
        return ju(a, start, i - 1) && ju(a, i, end - 1);
    }

    /**
     * 二叉树中和为某一值的路径
     */
    private ArrayList<ArrayList<Integer>> listAll = new ArrayList<ArrayList<Integer>>();
    private ArrayList<Integer> list = new ArrayList<Integer>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null) return listAll;
        list.add(root.val);
        target -= root.val;
        if (target == 0 && root.left == null && root.right == null)
            listAll.add(new ArrayList<Integer>(list));
        FindPath(root.left, target);
        FindPath(root.right, target);
        list.remove(list.size() - 1);
        return listAll;
    }

    /**
     * 复杂链表的复制
     */
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }

    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }

        RandomListNode currentNode = pHead;
        //复制每个节点，并插到被复制的节点后面
        while (currentNode != null) {
            RandomListNode cloneNode = new RandomListNode(currentNode.label);
            RandomListNode nextNode = currentNode.next;
            currentNode.next = cloneNode;
            cloneNode.next = nextNode;
            currentNode = nextNode;
        }

        currentNode = pHead;
        //重新遍历链表，复制老节点随机指针给新节点
        while (currentNode != null) {
            currentNode.next.random = currentNode.random == null ? null : currentNode.random.next;
            currentNode = currentNode.next.next;
        }

        //拆分链表
        currentNode = pHead;
        RandomListNode pCloneHead = pHead.next;
        while (currentNode != null) {
            RandomListNode cloneNode = currentNode.next;
            currentNode.next = cloneNode.next;
            cloneNode.next = cloneNode.next == null ? null : cloneNode.next.next;
            currentNode = currentNode.next;
        }

        return pCloneHead;
    }


    /**
     * 最小的K个数
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        int length = input.length;
        if (k > length || k == 0) {
            return result;
        }
        PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
        for (int i = 0; i < length; i++) {
            if (maxHeap.size() != k) {
                maxHeap.offer(input[i]);
            } else if (maxHeap.peek() > input[i]) {
                Integer temp = maxHeap.poll();
                temp = null;
                maxHeap.offer(input[i]);
            }
        }
        for (Integer integer : maxHeap) {
            result.add(integer);
        }
        return result;
    }

    /**
     * 二叉树的深度
     * @param root
     * @return
     */
    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int nLeft = TreeDepth(root.left);
        int nRight = TreeDepth(root.right);

        return (nLeft > nRight) ? (nLeft + 1) : (nRight + 1);
    }

    /**
     * 平衡二叉树
     * @param root
     * @return
     */
    public boolean IsBalanced_Solution(TreeNode root) {
        return getDepth(root) != -1;
    }

    private int getDepth(TreeNode root) {
        if (root == null) return 0;
        int left = getDepth(root.left);
        if (left == -1) return -1;
        int right = getDepth(root.right);
        if (right == -1) return -1;
        return Math.abs(left - right) > 1 ? -1 : 1 + Math.max(left, right);
    }

    /**
     * 数组中出现次数超过一半的数字
     * @param array
     * @return
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i : array) {
            if (map.containsKey(i)) {
                map.put(i, map.get(i) + 1);
            } else {
                map.put(i, 1);
            }
        }
        Iterator<Map.Entry<Integer, Integer>> iterator = map.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Integer, Integer> next = iterator.next();
            if (next.getValue() > array.length / 2) {
                return next.getKey();
            }
        }
        return 0;
    }

    /**
     * 数组中只出现一次的数字
     * @param array
     * @param num1
     * @param num2
     */
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i : array) {
            if (map.containsKey(i)) {
                map.put(i, map.get(i) + 1);
            } else {
                map.put(i, 1);
            }
        }
        Iterator<Map.Entry<Integer, Integer>> iterator = map.entrySet().iterator();
        boolean k = true;
        while (iterator.hasNext()) {
            Map.Entry<Integer, Integer> next = iterator.next();
            if (next.getValue() == 1) {
                if (k) {
                    num1[0] = next.getKey();
                    k = false;
                } else {
                    num2[0] = next.getKey();
                }
            }
        }
    }

    public char[] getNewString(char[] src, char[] find, char[] replace) {
        ArrayList<Character> res = new ArrayList<>();
        for (int i = 0; i < src.length; i++) {
            if (src[i] == find[0]) {
                boolean flag = true;
                for (int j = 1; j < find.length; j++) {
                    if (src[i + j] != find[j]) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    for (char c : replace) {
                        res.add(c);
                    }
                    i = i + find.length - 1;
                }
            } else {
                res.add(src[i]);
            }
        }
        char[] chars = new char[res.size()];
        int i = 0;
        for (Character r : res) {
            chars[i] = r;
        }
        return chars;
    }

    /**
     * 链表中环的入口结点
     */
    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        ListNode meetingNode = meetingNode(pHead);
        if (meetingNode == null) {
            return null;
        }
        // 得到环中的节点个数
        int nodeInLoop = 1;
        ListNode p1 = meetingNode;
        while (p1.next != meetingNode) {
            p1 = p1.next;
            ++nodeInLoop;
        }
        // 移动p1
        p1 = pHead;
        for (int i = 0; i < nodeInLoop; i++) {
            p1 = p1.next;
        }
        // 同时移动p1和p2
        ListNode p2 = pHead;
        while (p1 != p2) {
            p1 = p1.next;
            p2 = p2.next;
        }
        return p1;
    }

    //找到一快一满指针相遇处的节点，相遇的节点一定是在环中
    private ListNode meetingNode(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode slow = head.next;
        if (slow == null) {
            return null;
        }
        ListNode fast = slow.next;
        while (slow != null && fast != null) {
            if (slow == fast) {
                return fast;
            }
            slow = slow.next;
            fast = fast.next;
            if (fast != slow) {
                fast = fast.next;
            }
        }
        return null;
    }

    /**
     * 删除链表中的重复结点
     * @param pHead
     * @return
     */
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return pHead;
        }
        ListNode head = new ListNode(0);
        head.next = pHead;
        ListNode pre = head;
        ListNode last = head.next;
        while (last != null) {
            if (last.next != null && last.val == last.next.val) {
                // 找到最后的一个相同节点
                while (last.next != null && last.val == last.next.val) {
                    last = last.next;
                }
                pre.next = last.next;
                last = last.next;
            } else {
                pre = pre.next;
                last = last.next;
            }
        }
        return head.next;
    }

    /**
     * 剪绳子
     * @param target
     * @return
     */
    public int cutRope(int target) {
        if (target == 2) {
            return 1;
        }
        if (target == 3) {
            return 2;
        }
        int x = target % 3;
        int y = target / 3;
        if (x == 0) {
            return (int) Math.pow(3, y);
        } else if (x == 1) {
            return 2 * 2 * (int) Math.pow(3, y - 1);
        } else {
            return 2 * (int) Math.pow(3, y);
        }
    }

    /**
     * 机器人的运动范围
     * @param threshold
     * @param rows
     * @param cols
     * @return
     */
    public int movingCount(int threshold, int rows, int cols) {
        int flag[][] = new int[rows][cols];
        return helper(0, 0, rows, cols, flag, threshold);
    }

    private int helper(int i, int j, int rows, int cols, int[][] flag, int threshold) {
        if (i < 0 || i >= rows || j < 0 || j >= cols || numSum(i) + numSum(j) > threshold || flag[i][j] == 1)
            return 0;
        flag[i][j] = 1;
        return helper(i - 1, j, rows, cols, flag, threshold)
                + helper(i + 1, j, rows, cols, flag, threshold)
                + helper(i, j - 1, rows, cols, flag, threshold)
                + helper(i, j + 1, rows, cols, flag, threshold)
                + 1;
    }

    private int numSum(int i) {
        int sum = 0;
        do {
            sum += i % 10;
        } while ((i = i / 10) > 0);
        return sum;
    }

    /**
     * 连续子数组的最大和
     * @param array
     * @return
     */
    public int FindGreatestSumOfSubArray(int[] array) {
        if(array.length == 0 || array == null){
            return 0;
        }
        int curSum = 0;// 记录当前最大值
        int greatestSum = 0x80000000;// 初始化为最小值
        for (int i = 0;i < array.length;i++){
            curSum = (curSum < 0) ? array[i] : (curSum + array[i]);
//            if (curSum < 0){
//                curSum = array[i];// curSum是负数则舍弃，替换为当前值
//            } else {
//                curSum += array[i];// curSum为整数则加上当前值
//            }
            if (curSum > greatestSum){
                greatestSum = curSum;
            }
        }
        return greatestSum;
    }

    /**
     * 整数中1出现的次数
     * @param n
     * @return
     */
    public int NumberOf1Between1AndN_Solution(int n) {
        if (n <= 0){
            return 0;
        }
        int count = 0;
        for (long i = 1;i <= n;i *= 10){
            long diviver = i * 10;
            count += (n / diviver) * i + Math.min(Math.max(n % diviver - i + 1,0),i);
        }
        return count;
    }

    /**
     * 把数组排成最小的数
     * @param numbers
     * @return
     */
    public String PrintMinNumber(int [] numbers) {
        int length;
        String res = "";
        ArrayList<Integer> list = new ArrayList<>();
        length = numbers.length;
        for (int i = 0;i < length;i++){
            list.add(numbers[i]);
        }
        Collections.sort(list, new Comparator<Integer>() {
            @Override
            public int compare(Integer str1, Integer str2) {
                String s1 = str1 + "" + str2;
                String s2 = str2 + "" + str1;
                return s1.compareTo(s2);
            }
        });
        for (int j : list){
            res += j;
        }
        return res;
    }

    /**
     * 丑数
     * @param index
     * @return
     */
    public int GetUglyNumber_Solution(int index) {
        if (index == 0)
            return 0;
        int[] uglyNumbers = new int[index];
        uglyNumbers[0] = 1;
        int nextUglyIndex = 1;
        int index2 = 0,index3 = 0,index5 = 0;
        while (nextUglyIndex < index){
            int min = Math.min(Math.min(uglyNumbers[index2]*2,uglyNumbers[index3]*3),uglyNumbers[index5]*5);
            uglyNumbers[nextUglyIndex] = min;
            if (min == uglyNumbers[index2] * 2)
                index2++;
            if (min == uglyNumbers[index3] * 3)
                index3++;
            if (min == uglyNumbers[index5] * 5)
                index5++;
            nextUglyIndex++;
        }
        return uglyNumbers[nextUglyIndex - 1];
    }

    /**
     * 和为S的两个数字
     * @param array
     * @param sum
     * @return
     */
    public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
        ArrayList<Integer> resList = new ArrayList<>();
        if (array == null || array.length < 2){
            return resList;
        }
        int head = 0,behind = array.length - 1;
        while (head < behind){
            if (array[head] + array[behind] == sum){
                resList.add(array[head]);
                resList.add(array[behind]);
                return resList;
            } else if (array[head] + array[behind] < sum){
                head++;
            } else {
                behind--;
            }
        }
        return resList;
    }

    /**
     * 和为S的连续正数序列
     * @param sum
     * @return
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        int low = 1, high = 2;
        while (high > low) {
            int cur = (high + low) * (high - low + 1) / 2;
            if (cur == sum) {
                ArrayList<Integer> list = new ArrayList<>();
                for (int i = low; i <= high; i++) {
                    list.add(i);
                }
                result.add(list);
                low++;
            } else if (cur < sum) {
                high++;
            } else {
                low++;
            }
        }
        return result;
    }

    /**
     * 左旋转字符串
     * @param str
     * @param n
     * @return
     */
    public String LeftRotateString(String str,int n) {
        if (str == null || str.length() == 0)
            return str;
        int move = n % str.length();
        String str1 = str.substring(0,move);
        String str2 = str.substring(move,str.length());
        StringBuilder stringBuilder = new StringBuilder(str2);
        stringBuilder.append(str1);
        return stringBuilder.toString();
    }

    /**
     * 翻转单词顺序列
     * @param str
     * @return
     */
    public String ReverseSentence(String str) {
        if (str == null || str.trim().equals("")){
            return str;
        }
        String[] letters = str.split(" ");
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = letters.length - 1;i >= 0;i--){
            stringBuilder.append(letters[i]);
            if (i != 0){
                stringBuilder.append(" ");
            }
        }
        return stringBuilder.toString();
    }

    /**
     * 扑克牌顺子
     * @param numbers
     * @return
     */
    public boolean isContinuous(int [] numbers) {
        if (numbers.length != 5) return false;
        int[] d = new int[14];
        d[0] = -5;
        int len = numbers.length;
        int min = 14;
        int max = -1;
        for (int i = 0;i < len;i++){
            d[numbers[i]]++;
            if (numbers[i] == 0){
                continue;
            }
            if (d[numbers[i]] > 1){
                return false;
            }
            if (numbers[i] > max){
                max = numbers[i];
            }
            if (numbers[i] < min){
                min = numbers[i];
            }
        }
        if (max - min < 5){
            return true;
        }
        return false;
    }

    /**
     * 二叉搜索树与双向链表
     * @param pRootOfTree
     * @return
     */
    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null){
            return null;
        }
        if (pRootOfTree.left == null && pRootOfTree.right == null){
            return pRootOfTree;
        }
        // 将左子树构造成双链表，并返回链表头节点
        TreeNode left = Convert(pRootOfTree.left);
        TreeNode p = left;
        // 定位至左子树双链表最后一个节点
        while (p != null && p.right != null){
            p = p.right;
        }
        // 如果左子树链表不为空的话，将当前root追加到左子树链表
        if (left != null){
            p.right = pRootOfTree;
            pRootOfTree.left = p;
        }
        // 将右子树构造成双链表，并返回链表头节点
        TreeNode right = Convert(pRootOfTree.right);
        // 如果右子树链表不为空的话，将该链表追加到root节点之后
        if (right != null){
            right.left = pRootOfTree;
            pRootOfTree.right = right;
        }
        return left != null ? left : pRootOfTree;
    }

    /**
     * 求1+2+3+...+n
     * @param n
     * @return
     */
    public int Sum_Solution(int n) {
        int ans = n;
        if (ans != 0){
            ans += Sum_Solution(n - 1);
        }
        return ans;
    }

    /**
     * 不用加减乘除做加法
     * @param num1
     * @param num2
     * @return
     */
    public int Add(int num1,int num2) {
        int sum = num1 ^ num2;
        int carry = (num1 & num2) << 1;
        if (carry != 0){
            sum = Add(sum,carry);
        }
        return sum;
    }

    /**
     * 把字符串转换成整数
     * @param str
     * @return
     */
    public int StrToInt(String str) {
        int n = str.length(),s = 1;
        long res = 0;
        if (n == 0){
            return 0;
        }
        if (str.charAt(0) == '-'){
            s = -1;
        }
        for (int i = (str.charAt(0) == '-' || str.charAt(0) == '+') ? 1 : 0;i < n;i++){
            if (!(str.charAt(i) >= '0' && str.charAt(i) <= '9')){
                return 0;
            }
            res = res * 10 + str.charAt(i) - '0';
        }
        if (res * s > 2147483647 || res * s < -2147483648){
            return 0;
        }
        return (int)res * s;
    }

    /**
     * 数组中重复的数字
     * @param numbers
     * @param length
     * @param duplication
     * @return
     */
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        boolean[] k = new boolean[length];
        for (int i = 0;i < k.length;i++){
            if (k[numbers[i]] == true){
                duplication[0] = numbers[i];
                return true;
            }
            k[numbers[i]] = true;
        }
        return false;
    }

    /**
     * 构建乘积数组
     * @param A
     * @return
     */
    public int[] multiply(int[] A) {
        int length = A.length;
        int[] B = new int[length];
        if (length != 0){
            B[0] = 1;
            // 计算下三角
            for (int i = 1;i < length;i++){
                B[i] = B[i-1] * A[i - 1];
            }
            int temp = 1;
            // 计算上三角
            for (int j = length - 2;j >= 0;j--){
                temp *= A[j+1];
                B[j] *= temp;
            }
        }
        return B;
    }

    /**
     * 滑动窗口的最大值
     * @param num
     * @param size
     * @return
     */
    public ArrayList<Integer> maxInWindows(int [] num, int size) {
        ArrayList<Integer> integers = new ArrayList<>();
        if (num.length == 0 || size == 0)
            return integers;
        for (int i = 0;i + size - 1 < num.length;i++){
            int max = 0;
            for (int j = i;j < i + size;j++){
                max = (num[j] > max) ? num[j] : max;
            }
            integers.add(max);
        }
        return integers;
    }
}
