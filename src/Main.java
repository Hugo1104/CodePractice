import sun.awt.image.ImageWatched;

import java.util.*;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    public static void main(String[] args) {
        Set<String> set = new HashSet<>();

        set.add("hot");
        set.add("dot");
        set.add("dog");
        set.add("lot");
        set.add("log");


        findLadders("hit","cog", set);
    }

    public String longestPalindrome(String s) {
        // write your code here
        if (s == null || s.length() < 1){
            return "";
        }

        int start = 0;
        int end = 0;

        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCentre(s, i, i);
            int len2 = expandAroundCentre(s, i, i+1);
            int len = Math.max(len1, len2);
            if (len > end - start){
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);

    }

    public int expandAroundCentre(String s, int left, int right){
        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)){
            left--;
            right++;
        }
        return right - left - 1;
    }


    public int longestPalindromeSubseq(String s) {
        // write your code here
        if (s == null || s.length() < 1){
            return 0;
        }

        int n = s.length();
        int[][] dp = new int[n][n];

        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }

        for (int i = n - 1; i >= 0; i--) {
            char c1 = s.charAt(i);
            for (int j = i + 1; j < n; j++) {
                char c2 = s.charAt(j);
                if (c1 == c2) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                }else{
                    dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        return dp[0][n-1];
    }


    public int canCompleteCircuit(int[] gas, int[] cost) {
        // write your code here
        int n = gas.length;
        int i = 0;
        while (i < n) {
            int gasSum = 0, costSum = 0;
            int count = 0;
            while (count < n){
                int j = (i + count) % n;
                gasSum += gas[j];
                costSum += cost[j];
                if (gasSum < costSum){
                    break;
                }
                count++;
            }
            if (count == n){
                return i;
            }else{
                i += count + 1;
            }
        }
        return -1;
    }

    public static boolean validPalindrome(String s) {
        // Write your code here
        if (s == null){
            return false;
        }

        Pair pair = findDifference(s, 0, s.length() - 1);
        if (pair.left >= pair.right) {
            return true;
        }

        return isPalindrome(s, pair.left + 1, pair.right) || isPalindrome(s, pair.left, pair.right - 1);

    }

    public static Pair findDifference(String s, int left, int right){
        while (left < right && s.charAt(left) == s.charAt(right)){
            left++;
            right--;
        }

        return new Pair(left, right);
    }

    public static boolean isPalindrome(String s, int left, int right){
        Pair pair = findDifference(s, left, right);
        return pair.left >= pair.right;
    }

    public int[] twoSum(int[] numbers, int target) {
        // write your code here
        int[] result = new int[2];

        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                result[0] = map.get(target - numbers[i]);
                result[1] = i;
                return result;
            }
            map.put(numbers[i], i);
        }
        return new int[0];
    }

    public static int kthLargestElement(int k, int[] nums) {
        // write your code here
        return partition(nums, 0, nums.length - 1, nums.length - k);
    }

    public static int partition(int[] nums, int start, int end, int k) {

        int left = start, right = end;
        int pivot = nums[(start + end) / 2];

        while (left <= right) {
            while (left <= right && nums[left] < pivot){
                left++;
            }
            while (left <= right && nums[right] > pivot){
                right--;
            }

            if (left <= right) {
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
                left++;
                right--;
            }
        }

        if (left <= k){
            partition(nums, left, end, k);
        }

        if(right >= k){
            partition(nums, start, right, k);
        }

        return nums[k];
    }


    public int[] mergeSortedArray(int[] a, int[] b) {
        // write your code here
        if (a == null || b == null) {
            return null;
        }

        int indexA = 0;
        int indexB = 0;
        int index = 0;
        int[] result = new int[a.length + b.length];

        while (indexA < a.length && indexB < b.length) {
            if (a[indexA] <= b[indexB]) {
                result[index++] = a[indexA++];
            }else{
                result[index++] = b[indexB++];
            }
        }

        while(indexA < a.length){
            result[index++] = a[indexA++];
        }

        while(indexB < b.length){
            result[index++] = b[indexB++];
        }

        return result;
    }

    public void sortColors(int[] nums) {
        // write your code here
        int left = 0, index = 0, right = nums.length - 1;
        while (index <= right) {
            if (nums[index] == 0) {
                int temp = nums[index];
                nums[index] = nums[left];
                nums[left] = temp;
                index++;
                left++;
            }else if (nums[index] == 2) {
                int temp = nums[index];
                nums[index] = nums[right];
                nums[right] = temp;
                right--;
            }else {
                index++;
            }
        }
    }

    public void sortColorsQM(int[] nums) {
        sortQM(nums, 0, nums.length - 1);
    }

    public void sortQM(int[] nums, int start, int end) {
        if (start >= end) {
            return;
        }

        int left = start, right = end;
        int pivot = nums[(start + end) / 2];

        while (left <= right) {
            while (left <= right && nums[left] < pivot){
                left++;
            }
            while (left <= right && nums[right] > pivot){
                right--;
            }
            if (left <= right) {
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
                left++;
                right--;
            }
        }

        sortQM(nums, left, end);
        sortQM(nums, start, right);
    }

    public int fibonacci(int n) {
        // write your code here
        int a = 0;
        int b = 1;

        for (int i = 0; i < n - 1; i++) {
            int c = a + b;
            a = b;
            b = c;
        }

        return a;
    }

    public static List<Integer> minChairs(List<String> simulations) {
        // Write your code here
        List<Integer> result = new ArrayList<>();

        for (String simulation : simulations) {
            int total = 0, available = 0;
            for (int i = 0; i < simulation.length(); i++) {
                char action = simulation.charAt(i);

                if (action == 'C' || action == 'U') {
                    if (available == 0) {
                        total++;
                    }else{
                        available--;
                    }
                }

                if (action == 'R' || action == 'L') {
                    available++;
                }
            }
            result.add(total);
        }

        return result;
    }


    public static int countPalindromes(String s) {
        // Write your code here
        int counter = 0;

        for (int i = 0; i < s.length(); i++) {
            counter += palindromeCounter(s, i, i);
            counter += palindromeCounter(s, i, i + 1);
        }

        return counter;
    }

    public static int palindromeCounter(String s, int left, int right) {
        int counter = 0;

        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            counter++;
            left--;
            right++;
        }

        return counter;
    }

    public int findPosition(int[] nums, int target) {
        // write your code here
        if (nums.length < 1 || nums == null) {
            return -1;
        }

        int start = 0, end = nums.length - 1;

        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (nums[mid] == target) {
                return mid;
            }else if (nums[mid] < target) {
                start = mid;
            }else{
                end = mid;
            }
        }

        if (nums[start] == target) {
            return start;
        }

        if (nums[end] == target) {
            return end;
        }

        return -1;
    }

    public int mountainSequence(int[] nums) {
        // write your code here
        if (nums.length == 0 || nums == null) {
            return -1;
        }

        int start = 0, end = nums.length - 1;

        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (nums[mid] > nums[mid + 1]) {
                end = mid;
            }else{
                start = mid;
            }
        }

        return Math.max(nums[start], nums[end]);
    }

    public int partitionArray(int[] nums, int k) {
        // write your code here
        if (nums.length == 0) {
            return 0;
        }

        int left = 0, right = nums.length - 1;

        while (left <= right) {
            while (left <= right && nums[left] < k) {
                left++;
            }
            while (left <= right && nums[right] >= k) {
                right--;
            }

            if (left <= right) {
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;

                right--;
                left++;
            }
        }

        return left;
    }

    public List<List<Integer>> threeSum(int[] numbers) {
        // write your code here
        Arrays.sort(numbers);

        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < numbers.length; i++) {
            if (i != 0 && numbers[i] == numbers[i-1]) {
                continue;
            }

            findTwoSum(numbers, i, result);
        }

        return result;
    }

    public void findTwoSum(int[] numbers, int index, List<List<Integer>> result) {
        int left = index + 1, right = numbers.length - 1;
        int target = -numbers[index];

        while (left < right) {
            int twoSum = numbers[left] + numbers[right];
            if (twoSum < target) {
                left++;
            } else if (twoSum > target) {
                right--;
            } else {
                List<Integer> ans = new ArrayList<>();
                ans.add(numbers[index]);
                ans.add(numbers[left]);
                ans.add(numbers[right]);
                result.add(ans);
                left++;
                right--;
                while (left < right && numbers[left] == numbers[left - 1]) {
                    left++;
                }
            }
        }
    }

    public int subarraySumII(int[] a, int start, int end) {
        // write your code here
        int n = a.length;
        int ans = 0;

        if (end == 0) {
            return 0;
        }

        int sumStart = a[0], sumEnd = a[0];
        int leftStart = 0, leftEnd = 0;

        if (a[0] >= start && a[0] <= end) {
            ans++;
        }

        for (int i = 1; i < n; i++) {
            sumStart += a[i];
            sumEnd += a[i];

            while (sumStart > end) {
                sumStart -= a[leftStart++];
            }

            while (sumEnd - a[leftEnd] >= start) {
                sumEnd -= a[leftEnd++];
            }

            if (sumStart <= end && sumEnd >= start) {
                ans += leftEnd - leftStart + 1;
            }
        }

        return ans;
    }
    public int findPeak(int[] a) {
        // write your code here
        int start = 0, end = a.length - 2;

        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (a[mid] < a[mid + 1]) {
                start = mid;
            } else {
                end = mid;
            }
        }

        if(a[start] < a[end]) {
            return end;
        } else {
            return start;
        }
    }

    public int twoSum6(int[] nums, int target) {
        // write your code here
        if (nums == null || nums.length < 2) {
            return 0;
        }

        int count = 0;
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }

        for (int key : map.keySet()) {
            int diff = target - key;
            if ((diff == key && map.get(key) > 1) || (diff != key && map.containsKey(diff) && key < diff)) {
                count++;
            }
        }

        return count;
    }

    public int fourSumCount(int[] a, int[] b, int[] c, int[] d) {
        // Write your code here
        HashMap <Integer, Integer> map = new HashMap<>();
        int ans = 0;

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                int sum = a[i] + b[j];
                map.put(sum, map.getOrDefault(sum, 0) + 1);
            }
        }

        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < d.length; j++) {
                int target = -(c[i] + d[j]);
                if (map.containsKey(target)) {
                    ans += map.get(target);
                }
            }
        }

        return ans;
    }

    public TreeNode findSubtree(TreeNode root) {
        // write your code here
        Result result = subtreeSum(root);
        return result.node;
    }

    public Result subtreeSum (TreeNode node) {
        if (node == null) {
            return new Result(0, Integer.MIN_VALUE, null);
        }

        Result leftResult = subtreeSum(node.left);
        Result rightResult = subtreeSum(node.right);
        int sum = leftResult.sum + rightResult.sum + node.val;
        if (sum > Math.max(leftResult.maxSum, rightResult.maxSum)) {
            return new Result(sum, sum, node);
        } else if (leftResult.maxSum > rightResult.maxSum) {
            return new Result(sum, leftResult.maxSum, leftResult.node);
        } else {
            return new Result(sum, rightResult.maxSum, rightResult.node);
        }
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        // write your code here
        List result = new ArrayList();
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int size = queue.size();
            ArrayList<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode head = queue.poll();
                level.add(head.val);
                if (head.left != null) {
                    queue.offer(head.left);
                }
                if (head.right != null) {
                    queue.offer(head.right);
                }
            }
            result.add(level);
        }

        return result;
    }

    public List<List<Integer>> levelOrder2(TreeNode root) {
        // write your code here
        List result = new ArrayList();
        if (root == null) {
            return result;
        }

        List<TreeNode> queue = new ArrayList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            List<TreeNode> nextQueue = new ArrayList<>();
            result.add(toIntegerList(queue));
            for (TreeNode node : queue) {
                if (node.left != null) {
                    nextQueue.add(node.left);
                }
                if (node.right != null) {
                    nextQueue.add(node.right);
                }
            }
            queue = nextQueue;
        }

        return result;
    }

    public List<Integer> toIntegerList(List<TreeNode> list) {
        List<Integer> level = new ArrayList<>();
        for (TreeNode node : list) {
            level.add(node.val);
        }

        return level;
    }

    public List<List<Integer>> levelOrder3(TreeNode root) {
        // write your code here
        List result = new ArrayList();
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        queue.offer(null);

        List<Integer> level = new ArrayList<>();
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node == null) {
                if (level.size() == 0) {
                    break;
                }
                result.add(level);
                level = new ArrayList<>();
                queue.offer(null);
                continue;
            }
            level.add(node.val);
            if (node.left != null) {
                queue.add(node.left);
            }

            if (node.right != null) {
                queue.add(node.right);
            }
        }

        return result;
    }

    public int findCircleNum(int[][] m) {
        // Write your code here
        int n = m.length;
        int circle = 0;
        boolean[] visited = new boolean[n];

        for (int i = 0; i < n; i++) {
            visited[i] = false;
        }

        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                circle += 1;
                visited[i] = true;

                Queue<Integer> queue = new LinkedList<>();
                queue.add(i);

                while (!queue.isEmpty()) {
                    int now = queue.poll();
                    for (int j = 0; j < n; j++) {
                        if (visited[j] == false && m[now][j] == 1) {
                            visited[j] = true;
                            queue.offer(j);
                        }
                    }
                }
            }
        }

        return circle;

    }

    public int shortestPath2(boolean[][] grid) {
        // write your code here
        int R = grid.length - 1;
        int C = grid[0].length - 1;

        int[] dirRow = {1, -1, 2, -2};
        int[] dirCol = {2, 2, 1, 1};

        Point source = new Point(0, 0);
        Point destination = new Point(R, C);

        Queue<Point> queue = new LinkedList<>();
        queue.offer(source);

        int step = 0;

        while (!queue.isEmpty()) {
            step++;

            int currSize = queue.size();
            for (int i = 0; i < currSize; i++) {
                Point point = queue.poll();

                for (int j = 0; j < dirRow.length; j++) {
                    Point position = new Point(point.x + dirRow[j], point.y+dirCol[j]);

                    if (!inBound(position, R, C) || isBarrier(position, grid)) {
                        continue;
                    }

                    if (position.x == destination.x && position.y == destination.y) {
                        return step;
                    }

                    queue.offer(position);
                    grid[position.x][position.y] = true;
                }
            }
        }
        return -1;
    }

    boolean inBound(Point point, int R, int C) {
        int x = point.x;
        int y = point.y;

        if (x >= 0 && x <= R && y >= 0 && y <= C) {
            return true;
        }

        return false;
    }

    boolean isBarrier(Point point, boolean[][] grid) {
        if (grid[point.x][point.y]) {
            return true;
        }

        return false;
    }

    public int ladderLength(String start, String end, Set<String> dict) {
        // write your code here
        if (dict == null) {
            return 0;
        }
        if (start.equals(end)) {
            return 1;
        }

        dict.add(start);
        dict.add(end);

        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        queue.offer(start);
        visited.add(start);

        int count = 1;

        while (!queue.isEmpty()) {
            count++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String current = queue.poll();
                for (String word : getNext(current, dict)) {
                    if (visited.contains(word)) {
                        continue;
                    }
                    if (word.equals(end)) {
                        return count;
                    }
                    queue.add(word);
                    visited.add(word);
                }
            }
        }
        return 0;
    }

    ArrayList<String> getNext(String str, Set<String> dict) {
        ArrayList<String> result = new ArrayList<>();

        for (String possible : dict) {
            int diff = 0;
            for (int i = 0; i < possible.length(); i++) {
                if (possible.charAt(i) != str.charAt(i)) {
                    diff++;
                }
            }
            if (diff == 1) {
                result.add(possible);
            }
        }
        return result;
    }

    public boolean isBalanced(TreeNode root) {
        // write your code here
        if (root == null) {
            return true;
        } else {
            return (Math.abs(height(root.left) - height(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right));
        }
    }

    private int height (TreeNode node) {
        if (node == null) {
            return 0;
        } else {
            return Math.max(height(node.left), height(node.right)) + 1;
        }
    }

    public int maxDepth(TreeNode root) {
        // write your code here
        if (root == null) {
            return 0;
        }

        int leftHeight = maxDepth(root.left);
        int rightHeight = maxDepth(root.right);

        int height = Math.max(leftHeight, rightHeight) + 1;
        return height;
    }


    public double findMedianSortedArrays(int[] a, int[] b) {
        // write your code here
        double median;
        int totalLength = a.length + b.length;
        if (totalLength % 2 == 1) {
            int mid = totalLength / 2;
            median = getK(a, b ,mid + 1);
        } else {
            int left = totalLength / 2;
            int right = totalLength / 2 + 1;
            median = (getK(a, b, left) + getK(a, b, right)) / 2.0;
        }

        return median;
    }

    private int getK(int[] a, int[] b, int k) {
        int length1 = a.length, length2 = b.length;
        int index1 = 0, index2 = 0;
        int kElement = 0;

        while (true) {
            if (index1 == length1) {
                return b[index2 + k - 1];
            }

            if (index2 == length2) {
                return a[index1 + k - 1];
            }

            if (k == 1) {
                return Math.min(a[index1], b[index2]);
            }

            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, length1) - 1;
            int newIndex2 = Math.min(index2 + half, length2) - 1;
            int pivot1 = a[newIndex1];
            int pivot2 = b[newIndex2];

            if (pivot1 <= pivot2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }

    public int findMin(int[] nums) {
        // write your code here
        int start = 0, end = nums.length - 1;

        while (start < end) {
            int mid = (start + end) / 2;
            if (nums[end] > nums[mid]) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return nums[start];
    }


    public int[] kClosestNumbers(int[] a, int target, int k) {
        // write your code here
        int firstIndex = firstClosest(a, target);
        int left = firstIndex - 1, right = firstIndex;
        int[] result = new int[k];

        for (int i = 0; i < k; i++) {
            if (left < 0) {
                result[i] = a[right++];
            } else if (right == a.length) {
                result[i] = a[left--];
            } else {
                result[i] = target - a[left] <= a[right] - target ? a[left--] : a[right++];
            }
        }

        return result;

    }

    public int firstClosest (int[] a, int target) {
        int start = 0, end = a.length - 1;

        while (start + 1 < end) {
            int mid = (start + end) / 2;
            if (a[mid] < target) {
                start = mid;
            } else {
                end = mid;
            }
        }

        if (a[start] >= target) {
            return start;
        }

        if (a[end] >= target) {
            return end;
        }

        return a.length;
    }

    public double sqrt(double x) {
        // write your code here
        double start = 0, end = x;
        double mid;

        if (end < 1) {
            end = 1;
        }

        while (start + 1e-12 < end) {
            mid = (start + end) / 2;
            if (mid * mid < x) {
                start = mid;
            } else {
                end = mid;
            }
        }

        return start;
    }


    public ArrayList<DirectedGraphNode> topSort(ArrayList<DirectedGraphNode> graph) {
        // write your code here
        Map<DirectedGraphNode, Integer> map = new HashMap<>();
        Queue<DirectedGraphNode> queue = new LinkedList<>();
        ArrayList<DirectedGraphNode> result = new ArrayList<>();

        for (DirectedGraphNode node : graph) {
            for (DirectedGraphNode neighbor : node.neighbors) {
                map.put(neighbor, map.getOrDefault(neighbor, 0) + 1);
            }
        }

        for (DirectedGraphNode node : graph) {
            if (map.containsKey(node)) {
                continue;
            }
            queue.offer(node);
            result.add(node);
        }

        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                DirectedGraphNode node = queue.poll();
                for (DirectedGraphNode neighbor : node.neighbors) {
                    map.put(neighbor, map.get(neighbor) - 1);
                    if (map.get(neighbor) == 0) {
                        queue.offer(neighbor);
                        result.add(neighbor);
                    }
                }
            }
        }
        return result;
    }


    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        // write your code here
        if (node == null) {
            return null;
        }

        Map<UndirectedGraphNode, UndirectedGraphNode> visited = new HashMap<>();
        Queue<UndirectedGraphNode> queue = new LinkedList<>();

        visited.put(node, new UndirectedGraphNode(node.label));
        queue.offer(node);

        while (!queue.isEmpty()) {
            UndirectedGraphNode n = queue.poll();
            for (UndirectedGraphNode neighbor : n.neighbors) {
                if (!visited.containsKey(neighbor)) {
                    visited.put(neighbor, new UndirectedGraphNode(neighbor.label));
                    queue.add(neighbor);
                }
                visited.get(n).neighbors.add(visited.get(neighbor));
            }
        }

        return visited.get(node);
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        // write your code here
        List<List<Integer>> results = new ArrayList<>();
        ArrayList<Integer> indexes = new ArrayList<>();

        Arrays.sort(nums);
        results.add(new ArrayList<Integer>());
        indexes.add(-1);

        for (int i = 0; i < nums.length; i++) {
            int size = results.size();
            for (int j = 0; j < size; j++) {
                if (i > 0 && nums[i] == nums[i - 1] && indexes.get(j) != i - 1) {
                    continue;
                }

                results.add(new ArrayList<>(results.get(j)));
                results.get(results.size() - 1).add(nums[i]);
                indexes.add(i);
            }
        }
        return results;
    }

    public void moveZeroes(int[] nums) {
        // write your code here
        if (nums == null || nums.length == 0) {
            return;
        }

        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                int temp = nums[i];
                nums[i] = nums[index];
                nums[index++] = temp;
            }
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        // write your code here
        int len = nums.length;
        List<List<Integer>> results = new ArrayList<>();

        if (nums.length == 0) {
            results.add(new ArrayList<>());
            return results;
        }

        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);

        dfsPermute(nums, used, path, results);
        return results;
    }

    private void dfsPermute(int[] nums, boolean[] used, Deque<Integer> path, List<List<Integer>> results) {
        if (path.size() == nums.length) {
            results.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                path.addLast(nums[i]);
                used[i] = true;

                dfsPermute(nums, used, path, results);

                path.removeLast();
                used[i] = false;
            }
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        // write your code here
        int len = nums.length;
        List<List<Integer>> results = new ArrayList<>();

        if (nums.length == 0) {
            results.add(new ArrayList<>());
            return results;
        }

        Arrays.sort(nums);

        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);

        dfsDupPermute(nums, used, path, results);
        return results;

    }

    private void dfsDupPermute(int[] nums, boolean[] used, Deque<Integer> path, List<List<Integer>> results) {
        if (path.size() == nums.length) {
            results.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }

            if (i != 0 && nums[i] == nums[i - 1] && used[i - 1]) {
                continue;
            }

            used[i] = true;
            path.addLast(nums[i]);
            dfsDupPermute(nums, used, path, results);
            used[i] = false;
            path.removeLast();
        }
    }

    public int minCost(int n, int[][] roads) {
        // Write your code here
        int[][] graph = constructGraph(roads, n);
        Set<Integer> visited = new HashSet<>();
        dfsResult result = new dfsResult();
        visited.add(1);

        minCostdfs(1, n, visited, 0, graph, result);
        return result.minCost;
    }

    void minCostdfs(int city,
                    int n,
                    Set<Integer> visited,
                    int cost,
                    int[][] graph,
                    dfsResult result) {
        if (visited.size() == n) {
            result.minCost = Math.min(result.minCost, cost);
            return;
        }

        for (int i = 1; i < graph[city].length; i++) {
            if (visited.contains(i)) {
                continue;
            }

            visited.add(i);
            minCostdfs(i, n, visited,cost + graph[city][i], graph, result);
            visited.remove(i);
        }
    }

    int[][] constructGraph(int[][] roads, int n) {
        int[][] graph = new int[n + 1][n + 1];
        for (int i = 0; i < n + 1; i++) {
            for (int j = 0; j < n + 1; j++) {
                graph[i][j] = Integer.MAX_VALUE >> 4;
            }
        }

        for (int i = 0; i < roads.length; i++) {
            int a = roads[i][0];
            int b = roads[i][1];
            int c = roads[i][2];
            graph[a][b] = Math.min(graph[a][b], c);
            graph[b][a] = Math.min(graph[b][a], c);
        }

        return graph;
    }


    public int hashCode(String key, int HASH_SIZE) {
        // write your code here
        long result = 0;

        for (int i = 0; i < key.length(); i++) {
            result = (result * 33 + key.charAt(i)) % HASH_SIZE;
        }

        return (int) result;
    }


    public ListNode[] rehashing(ListNode[] hashTable) {
        // write your code here
        int cap = hashTable.length * 2;

        ListNode[] table = new ListNode[cap];
        ListNode[] tail = new ListNode[cap];

        for (int i = 0; i < hashTable.length; i++) {
            while (hashTable[i] != null) {
                ListNode current = hashTable[i];
                int hashCode = (current.val + cap) % cap;
                hashTable[i] = current.next;
                if (tail[hashCode] != null) {
                    tail[hashCode].next = current;
                } else {
                    table[hashCode] = current;
                }

                tail[hashCode] = current;
                current.next = null;
            }
        }

        return table;
    }

    public int nthUglyNumber(int n) {
        // write your code here
        int[] factors = {2, 3 ,5};
        Set<Long> seen = new HashSet<>();
        PriorityQueue<Long> list = new PriorityQueue<>();
        int ugly = 0;
        seen.add(1L);
        list.add(1L);

        for (int i = 0; i < n; i++) {
            long current = list.poll();
            ugly = (int) current;

            for (int factor : factors) {
                long newNum = current * factor;
                if (seen.add(newNum)) {
                    list.offer(newNum);
                }
            }
        }

        return ugly;
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // write your code here
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> combine = new ArrayList<>();
        combineDFS(candidates, target, combine, results, 0);
        return results;
    }

    private void combineDFS(int[] candidates,
                            int target,
                            List<Integer> combine,
                            List<List<Integer>> results,
                            int index) {
        if (index == candidates.length) {
            return;
        }

        if (target == 0) {
            List<Integer> ans = new ArrayList<>(combine);
            Collections.sort(ans);
            if (!results.contains(ans)) {
                results.add(ans);
            }
            return;
        }

        combineDFS(candidates, target, combine, results, index + 1);
        if (target - candidates[index] >= 0) {
            combine.add(candidates[index]);
            combineDFS(candidates, target - candidates[index], combine, results, index);
            combine.remove(combine.size() - 1);
        }
    }

    public List<String> stringPermutation2(String str) {
        // write your code here
        List<String> results = new ArrayList<>();

        if (str == null) {
            return results;
        }

        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        boolean[] visited = new boolean[chars.length];

        strPermutationDFS(chars, "", visited, results);
        return results;
    }

    private void strPermutationDFS(char[] chars, String permutation, boolean[] visited, List<String> results) {
        if (permutation.length() == chars.length) {
            results.add(permutation);
            return;
        }

        for (int i = 0; i < chars.length; i++) {
            if (visited[i]) {
                continue;
            }

            if (i > 0 && chars[i - 1] == chars[i] && !visited[i - 1]) {
                continue;
            }

            visited[i] = true;
            strPermutationDFS(chars, permutation + chars[i], visited, results);
            visited[i] = false;
        }
    }

    public int kthSmallest(TreeNode root, int k) {
        // write your code here
        ArrayDeque<TreeNode> stack = new ArrayDeque<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.offer(root);
                root = root.left;
            }
            root = stack.poll();
            --k;
            if (k == 0) {
                break;
            }
            root = root.right;
        }

        return root.val;
    }

    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;

        while (i >= 0 && nums[i + 1] <= nums[i]) {
            i--;
        }

        if (i >= 0) {
            int j = nums.length - 1;
            while (nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    public List<List<Integer>> kSumII(int[] a, int k, int target) {
        // write your code here
        List<List<Integer>> results = new ArrayList<>();
        if (a.length < 1) {
            return results;
        }
        Arrays.sort(a);
        List<Integer> result = new ArrayList<>();
        ksumIIDFS(a, k, target, 0, result, results);
        return results;
    }
    
    private void ksumIIDFS(int[] a,
                           int k,
                           int target,
                           int index,
                           List<Integer> result,
                           List<List<Integer>> results) {
        if (result.size() == k && target == 0) {
            results.add(new ArrayList<>(result));
            return;
        }

        if (target == 0) {
            return;
        }

        for (int i = index; i < a.length; i++) {
            if (a[i] > target) {
                break;
            }
            result.add(a[i]);
            ksumIIDFS(a, k, target - a[i], i + 1, result, results);
            result.remove(result.size() - 1);
        }
    }


    public List<String> letterCombinations(String digits) {
        // write your code here
        List<String> results = new ArrayList<>();
        if (digits.length() == 0) {
            return results;
        }

        Map<Character, String> map = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};

        StringBuffer str = new StringBuffer();
        letterCombinationDFS(digits, map, 0, str, results);
        return results;
    }

    private void letterCombinationDFS(String digits,
                                      Map<Character, String> map,
                                      int index,
                                      StringBuffer ans,
                                      List<String> results) {
        if (index == digits.length()) {
            results.add(ans.toString());
            return;
        }

        char digit = digits.charAt(index);
        String letters = map.get(digit);
        for (int i = 0; i < letters.length(); i++) {
            ans.append(letters.charAt(i));
            letterCombinationDFS(digits, map, index + 1, ans, results);
            ans.deleteCharAt(ans.length() - 1);
        }
    }

    public int minimumTotal(int[][] triangle) {
        // write your code here
        boolean[][] visited = new boolean[triangle.length][];
        int[][] memo = new int[triangle.length][];

        for (int i = 0; i < triangle.length; i++) {
            visited[i] = new boolean[triangle[i].length];
            memo[i] = new int[triangle[i].length];
        }

        return triangleHelper(triangle, 0, 0, visited, memo);
    }

    public int triangleHelper(int[][] triangle, int row, int column, boolean[][] visited, int[][] memo) {
        if (visited[row][column]) {
            return memo[row][column];
        }

        int number = triangle[row][column];

        if (row + 1 == triangle.length) {
            return number;
        }

        int leftNum = triangleHelper(triangle, row + 1, column, visited, memo);
        int rightNum = triangleHelper(triangle, row + 1, column + 1, visited, memo);
        int result = 0;

        if (leftNum < rightNum) {
            result = number + leftNum;
        } else {
            result = number + rightNum;
        }

        visited[row][column] = true;
        memo[row][column] = result;

        return result;

    }

    public boolean canWinBash(int n) {
        return n % 4 != 0;
    }


    public static void mergeSortedArray(int[] A, int m, int[] B, int n) {
        // write your code here
        int[] result = new int[m + n];
        int i = 0, j = 0;
        while (i < m && j < n) {
            if (A[i] <= B[j]) {
                result[i + j] = A[i];
                i++;
            } else {
                result[i + j] = B[j];
                j++;
            }
        }

        while (i < m) {
            result[i + j] = A[i];
            i++;
        }

        while (j < n) {
            result[i + j] = B[j];
            j++;
        }

        for (int k = 0; k < m + n; k++) {
            A[k] = result[k];
        }
    }

    public void invertBinaryTree(TreeNode root) {
        // write your code here
        if (root == null) {
            return;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            TreeNode node =  queue.poll();
            TreeNode temp = node.left;
            node.left = node.right;
            node.right = temp;

            if (node.right != null) {
                queue.offer(node.right);
            }
            if (node.left != null) {
                queue.offer(node.left);
            }
        }
    }

    public List<String> wordSearchII(char[][] board, List<String> words) {
        // write your code here
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }

        Set<String> ans = new HashSet<String>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                wordSearchIIDFS(board, trie, i, j, ans, dirs);
            }
        }

        return new ArrayList<String>(ans);
    }

    private void wordSearchIIDFS(char[][] board, Trie trie, int i, int j, Set<String> ans, int[][] dirs) {
        if (!trie.children.containsKey(board[i][j])) {
            return;
        }

        char c = board[i][j];
        Trie next = trie.children.get(c);

        if (!"".equals(next.word)) {
            ans.add(next.word);
            next.word = "";
        }

        if (!next.children.isEmpty()) {
            board[i][j] = '#';
            for (int[] dir : dirs) {
                int x = i + dir[0], y = j + dir[1];
                if (x >= 0 && x < board.length && y >= 0 && y < board[0].length) {
                    wordSearchIIDFS(board, next, x, y, ans, dirs);
                }
            }
            board[i][j] = c;
        }

        if (next.children.isEmpty()) {
            trie.children.remove(c);
        }
    }

    class Trie {
        String word;
        Map<Character, Trie> children;
        boolean isWord;

        public Trie() {
            this.word = "";
            this.children = new HashMap<Character, Trie>();
        }

        public void insert(String word) {
            Trie cur = this;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (!cur.children.containsKey(c)) {
                    cur.children.put(c, new Trie());
                }
                cur = cur.children.get(c);
            }
            cur.word = word;
        }
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        // write your code here
        if (obstacleGrid.length == 0 || obstacleGrid == null) {
            return 0;
        }

        int n = obstacleGrid.length, m = obstacleGrid[0].length;
        int[][] dp = new int[n][m];

        for (int i = 0; i < n; i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            }
            dp[i][0] = 1;
        }

        for (int j = 0; j < m; j++) {
            if (obstacleGrid[0][j] == 1) {
                break;
            }
            dp[0][j] = 1;
        }

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                if (obstacleGrid[i][j] == 1) {
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[n - 1][m - 1];
    }

    public int shortestPath2DP(boolean[][] grid) {
        // write your code here
        int[] deltaX = {1, -1, 2, -2};
        int[] deltaY = {-2, -2, -1, -1};

        if (grid.length == 0 || grid == null) {
            return -1;
        }

        int n = grid.length, m = grid[0].length;
        int[][] dp = new int[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                dp[i][j] = Integer.MAX_VALUE;
            }
        }

        dp[0][0] = 0;

        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                if (grid[i][j]) {
                    continue;
                }

                for (int direction = 0; direction < 4; direction++) {
                    int x = i + deltaX[direction];
                    int y = j + deltaY[direction];
                    if (x < 0 || x >= n || y < 0 || y >= m) {
                        continue;
                    }
                    if (dp[x][y] == Integer.MAX_VALUE) {
                        continue;
                    }

                    dp[i][j] = Math.min(dp[i][j], dp[x][y] + 1);
                }
            }
        }

        return dp[n - 1][m - 1] == Integer.MAX_VALUE ? -1 : dp[n - 1][m - 1];
    }

    public boolean canJump(int[] a) {
        // write your code here
        if (a == null) {
            return false;
        }

        int n = a.length;
        boolean[] dp = new boolean[n];
        dp[0] = true;

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && a[j] + j >= i) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n - 1];

    }

    public int backPack(int m, int[] a) {
        // write your code here
        int n = a.length;

        boolean[][] dp = new boolean[n + 1][m + 1];

        dp[0][0] = true;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= a[i - 1]) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j - a[i - 1]];
                }
            }
        }

        for (int i = m; i >= 0 ; i--) {
            if (dp[n][i]) {
                return i;
            }
        }
        return 0;
    }

    private class DataStream {
        private class Node {
            public Node next;
            public int value;
            public Node(int value) {
                this.value = value;
                this.next = null;
            }
        }

        private HashMap<Integer, Node> hashmap;
        private Node dummpy;
        private Node cur;
        public DataStream() {
            hashmap = new HashMap<Integer, Node>();
            dummpy = new Node(0);
            cur = dummpy;
        }

        public void add(int value) {
            if(!hashmap.containsKey(value)) {
                Node node = new Node(value);
                cur.next = node;
                hashmap.put(value, cur);
                cur = cur.next;
            } else {
                // contains duplicate key;
                Node prev = hashmap.get(value);
                if(prev != null) {
                    prev.next = prev.next.next;
                    if(prev.next != null) {
                        int prevNextValue = prev.next.value;
                        hashmap.put(prevNextValue, prev);
                    } else {
                        //这里很重要，别忘记了，如果delete掉是最后一个元素，那么cur = prev;
                        cur = prev;
                    }
                    //这里很巧妙的,如果是第二次遇见，那么直接设置node为null,
                    //但是hashmap里面还是保留key，表示遇见过；
                    hashmap.put(value, null);
                }
            }
        }

        public int getFirstUnique() {
            if(dummpy.next != null) {
                return dummpy.next.value;
            } else {
                return -1;
            }
        }
    }

    public int firstUniqueNumber(int[] nums, int number) {
        if(nums == null || nums.length == 0) {
            return -1;
        }
        DataStream ds = new DataStream();
        for(int i = 0; i < nums.length; i++) {
            ds.add(nums[i]);
            if(nums[i] == number){
                return ds.getFirstUnique();
            }
        }
        return -1;
    }


    public int backPackII(int m, int[] a, int[] v) {
        // write your code here
        int n = a.length;
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 0;
                } else if (a[i - 1] > j ) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - a[i - 1]] + v[i - 1]);
                }
            }
        }

        return dp[n][m];
    }


    public int backPackV(int[] nums, int target) {
        // write your code here
        int n = nums.length;

        if (n == 0) {
            return 0;
        }

        int[][] dp = new int[n + 2][target + 2];

        for (int i = 0; i <= target; i++) {
            if (i == 0) {
                dp[0][i] = 1;
            } else {
                dp[0][i] = 0;
            }
        }

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= target; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= nums[i - 1]) {
                    dp[i][j] += dp[i - 1][j - nums[i - 1]];
                }
            }
        }

        return dp[n][target];
    }


    public int wordBreak3(String s, Set<String> dict) {
        // Write your code here
        if (s.length() == 0 || dict.size() == 0) {
            return 0;
        }

        s = s.toLowerCase();
        Set<String> set = new HashSet<>();
        for (String str : dict) {
            set.add(str.toLowerCase());
        }

        int len = s.length();
        int[] dp = new int[len + 1];
        
        dp[0] = 1;

        for (int i = 0; i < len; i++) {
            for (int j = i; j < len; j++) {
                if (set.contains(s.substring(i, j + 1))) {
                    dp[j + 1] += dp[i];
                }
            }
        }

        return dp[len];
    }

    public int climbStairs2(int n) {
        // write your code here
        if (n <= 1) {
            return 1;
        }

        if (n == 2) {
            return 2;
        }

        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;

        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3];
        }

        return dp[n];
    }


    public int[] winSum(int[] nums, int k) {
        // write your code here
        if (k == 0) {
            return new int[0];
        }

        int n = nums.length;
        int[] results = new int[n - k + 1];

        int left = 0, right = k - 1, sum = 0;
        for (int i = 0; i <= right; i++) {
            sum += nums[i];
        }

        int index = 0;
        results[index++] = sum;

        while (right < n - 1) {
            sum -= nums[left];
            left++;
            right++;
            sum += nums[right];
            results[index++] = sum;
        }
        return results;
    }

    public int gcd(int a, int b) {
        // write your code here
        int smaller = Math.min(a, b);
        int gcd = 0;
        for (int i = 1; i <= smaller; i++) {
            if (a % i == 0 && b % i == 0) {
                gcd = i;
            }
        }
        return gcd;
    }

    public int longestIncreasingSubsequence(int[] nums) {
        // write your code here
        if (nums.length == 0) {
            return 0;
        }

        int[] dp = new int[nums.length];
        dp[0] = 1;
        int ans = 1;

        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            ans = Math.max(ans, dp[i]);
        } 
        return ans;
    }

    public List<Interval> mergeKSortedIntervalLists(List<List<Interval>> intervals) {
        // write your code here
        if (intervals.size() == 0) {
            return new ArrayList<>();
        }

        List<Interval> list = new ArrayList<>();

        for (List<Interval> interval : intervals) {
            list.addAll(interval);
        }

        list.sort((Interval a, Interval b) -> a.start - b.start);
        List<Interval> ans = new ArrayList<>();
        ans.add(list.get(0));
        for (int i = 1; i < list.size(); i++) {
            int lastPos = ans.size() - 1;
            if (ans.get(lastPos).end >= list.get(i).start) {
                ans.get(lastPos).end = Math.max(ans.get(lastPos).end, list.get(i).end);
            } else {
                ans.add(list.get(i));
            }
        }
        return ans;
    }

    public List<Integer> largestDivisibleSubset(int[] nums) {
        // write your code here
        if (nums.length == 0) {
            return new ArrayList<>();
        }

        Arrays.sort(nums);

        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] dp = new int[nums.length];
        int globalMaxIndex = 0, globalMaxCount = 0;

        for (int i = 0; i < nums.length; i++) {
            map.put(i, new ArrayList<>());
            int preIndex = 0;
            for (int j = 0; j < i; j++) {
                if (nums[i] % nums[j] == 0) {
                    if (dp[i] < dp[j] + 1) {
                        dp[i] = dp[j] + 1;
                        preIndex = j;
                    }
                }
            }

            if (dp[i] != 0) {
                map.get(i).addAll(map.get(preIndex));
            }
            map.get(i).add(nums[i]);
            if (dp[i] > globalMaxCount) {
                globalMaxCount = dp[i];
                globalMaxIndex = i;
            }
        }

        return map.get(globalMaxIndex);
    }

    public int deduplication(int[] nums) {
        // write your code here
        if (nums.length == 0) {
            return 0;
        }

        Arrays.sort(nums);
        int first = 0;
        for (int second = 0; second < nums.length; second++) {
            if (second == 0 || nums[second] != nums[second - 1]) {
                nums[first++] = nums[second];
            }
        }

        return first;
    }

    public String sumofTwoStrings(String a, String b) {
        // write your code here
        if (a.length() == 0) {
            return b;
        }

        if (b.length() == 0) {
             return a;
        }

        String ans = new String();
        if (a.length() <= b.length()) {
             ans = twoStringHelper(a, b).toString();
        } else {
             ans = twoStringHelper(b, a).toString();
        }

        return ans;
    }

    private StringBuilder twoStringHelper(String shorter, String longer) {
        int index1 = 0, index2 = 0;

        StringBuilder ans = new StringBuilder();
        while (index2 < longer.length() - shorter.length()) {
            ans.append(longer.charAt(index2++));
        }

        while (index1 < shorter.length()) {
            int value = Character.getNumericValue(shorter.charAt(index1++)) + Character.getNumericValue(longer.charAt(index2++));
            ans.append(value);
        }

        return ans;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        // write your code here
        if (matrix.length == 0) {
            return false;
        }

        int m = matrix.length, n = matrix[0].length;
        int low = 0, high = m * n - 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            int num = matrix[mid / n][mid % n];

            if (num < target) {
                low = mid + 1;
            } else if (num > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }

        return false;
    }

    public int totalOccurrence(int[] a, int target) {
        // write your code here
        if (a.length == 0) {
            return 0;
        }

        int left = 0, right = a.length - 1;

        while (left + 1 < right) {
            int mid = (left + right) / 2;
            if (a[mid] < target) {
                left = mid;
            } else {
                right = mid;
            }
        }

        int startIndex = 0;
        if (a[left] == target) {
            startIndex = left;
        } else if (a[right] == target) {
            startIndex = right;
        } else {
            return 0;
        }

        left = 0;
        right = a.length - 1;

        while (left + 1 < right) {
            int mid = (left + right) / 2;
            if (a[mid] > target) {
                right = mid;
            } else {
                left = mid;
            }
        }

        int endIndex = 0;
        if (a[right] == target) {
            endIndex = right;
        } else if (a[left] == target) {
            endIndex = left;
        }

        return endIndex - startIndex + 1;
    }

    public int threeSumClosest(int[] numbers, int target) {
        // write your code here
        Arrays.sort(numbers);
        int ans = numbers[0] + numbers[1] + numbers[2];

        for (int i = 0; i < numbers.length; i++) {
            int left = i + 1, right = numbers.length - 1;
            while (left < right) {
                int sum = numbers[i] + numbers[left] + numbers[right];
                if (Math.abs(target - sum) < Math.abs(target - ans)) {
                    ans = sum;
                }

                if (sum > target) {
                    right--;
                } else if (sum < target) {
                    left++;
                } else {
                    return ans;
                }
            }
        }
        return ans;
    }

    public void sortColors2(int[] colors, int k) {
        // write your code here
        int start = 0;
        for (int i = 1; i < k; i++) {
            int pointer = start;
            while (pointer < colors.length) {
                if (colors[pointer] != i) {
                    pointer++;
                    continue;
                }
                int temp = colors[start];
                colors[start++] = colors[pointer];
                colors[pointer++] = temp;
            }
        }
    }


    public int[] twoSum7(int[] nums, int target) {
        // write your code here
        if (nums.length == 0) {
            return new int[]{-1, -1};
        }

        target = Math.abs(target);
        int j = 1;

        for (int i = 0; i < nums.length; i++) {
            j = Math.max(i + 1, j);
            while (j < nums.length && nums[j] - nums[i] < target) {
                j++;
            }

            if (j == nums.length) {
                break;
            }

            if (nums[j] - nums[i] == target) {
                return new int[]{nums[i], nums[j]};
            }
        }

        return new int[]{-1, -1};
    }

    public int stringCount(String str) {
        // Write your code here.
        if (str.length() == 0) {
            return 0;
        }

        int j = 1, ans = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) != '0') {
                continue;
            }

            j = Math.max(i + 1, j);
            while (j < str.length() && str.charAt(j) == '0') {
                j++;
            }

            ans += j - i;
        }

        return ans;
    }

    public int deduplication2(int[] nums) {
        // write your code here
        if (nums.length == 0) {
            return 0;
        }

        Arrays.sort(nums);

        int i, j = 1;
        for (i = 0; i < nums.length; i++) {
            j = Math.max(i + 1, j);
            while (j < nums.length && nums[i] == nums[j]) {
                j++;
            }

            if (j >= nums.length) {
                break;
            }

            nums[i + 1] = nums[j];
        }

        return i + 1;
    }

    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        // write your code here
        if (s.length() == 0 || k == 0) {
            return 0;
        }

        int left = 0, right = 0, count = 0;
        int[] charSet = new int[256];
        int ans = 0;

        while (right < s.length()) {
            if (charSet[s.charAt(right)] == 0) {
                count++;
            }

            charSet[s.charAt(right)]++;
            right++;

            while (count > k) {
                charSet[s.charAt(left)]--;
                if (charSet[s.charAt(left)] == 0) {
                    count--;
                }
                left++;
            }

            ans = Math.max(ans, right - left);
        }

        return ans;
    }

    public int[] winSumReview(int[] nums, int k) {
        // write your code here
        if (nums.length == 0 || k == 0) {
            return new int[]{};
        }

        int[] ans = new int[nums.length - k + 1];

        int j = 0, sum = 0;
        for (int i = 0; i < nums.length; i++) {
            while (j < nums.length && j - i < k) {
                sum += nums[j];
                j++;
            }
            if (j - i == k) {
                ans[i] = sum;
            }
            sum -= nums[i];
        }

        return ans;
    }

    public int characterReplacement(String s, int k) {
        // write your code here
        if (s.length() == 0) {
            return 0;
        }

        int j = 0, ans = 0, maxFreq = 0, count;
        HashMap<Character, Integer> counter = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            while (j < s.length() && j - i - maxFreq <= k) {
                count = counter.getOrDefault(s.charAt(j), 0) + 1;
                counter.put(s.charAt(j), count);
                maxFreq = Math.max(maxFreq, count);
                j++;
            }

            if (j - i - maxFreq > k) {
                ans = Math.max(ans, j - i - 1);
            } else {
                ans = Math.max(ans, j - i);
            }

            count = counter.get(s.charAt(i)) - 1;
            counter.put(s.charAt(i), count);
            maxFreq = getMaxFreq(counter);
        }

        return ans;
    }

    private int getMaxFreq(HashMap<Character, Integer> map) {
        int result = 0;
        for (Integer num : map.values()) {
            result = Math.max(result, num);
        }

        return result;
    }

    public boolean hasCycle(ListNode head) {
        // write your code here
        if (head == null || head.next == null) {
            return false;
        }

        ListNode fast, slow;
        fast = head.next;
        slow = head;

        while (fast != slow) {
            if (fast == null || fast.next == null) {
                return false;
            }
            fast = fast.next.next;
            slow = slow.next;
        }

        return true;
    }

        public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            // write your code here
            Set<ListNode> set = new HashSet<>();
            ListNode temp = headA;
            while (temp != null) {
                set.add(temp);
                temp = temp.next;
            }
            temp = headB;
            while (temp != null) {
                if (set.contains(temp)) {
                    return temp;
                }
                temp = temp.next;
            }

            return null;
        }

    public ListNode middleNode(ListNode head) {
        // write your code here
        if (head == null) {
            return null;
        }

        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        return slow;
    }

    public boolean searchMatrixReview(int[][] matrix, int target) {
        // write your code here
        if (matrix.length == 0) {
            return false;
        }

        int m = matrix.length, n = matrix[0].length;
        int low = 0, high = m * n - 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            int num = matrix[mid / n][mid % n];

            if (num < target) {
                low = mid + 1;
            } else if (num > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }

    public int searchMatrixII(int[][] matrix, int target) {
        // write your code here
        if (matrix.length == 0) {
            return 0;
        }

        int i = matrix.length - 1, j = 0;
        int counter = 0;

        while (i >= 0 && j < matrix[0].length) {
            int num = matrix[i][j];

            if (num < target) {
                j++;
            } else if (num > target) {
                i--;
            } else {
                i--;
                j++;
                counter++;
            }
        }
        return counter;
    }

    public int minArea(char[][] image, int x, int y) {

        if (image == null || image.length == 0 || image[0].length == 0) {
            return 0;
        }



        int n = image.length;
        int m = image[0].length;
        int l = 0, r = 0;
        int left = 0, right = 0, top = 0, bottom = 0;

        //二分最左侧黑色像素坐标
        l = 0;
        r = y;
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;

            if (check_column(image, mid)) {
                r = mid;
            } else {
                l = mid;
            }
        }

        if (check_column(image, l)){
            left = l;

        }else{
            left = r;
        }

        //二分最右侧黑色像素坐标
        l = y;
        r = m - 1;

        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            if (check_column(image, mid)) {
                l = mid;
            } else {
                r = mid;
            }
        }

        if (check_column(image, r)) {
            right = r;
        }else{
            right = l;
        }

        //二分最上侧黑色像素坐标
        l = 0;
        r = x;
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            if (check_row(image, mid, left, right)) {
                r = mid;
            } else {
                l = mid;
            }
        }

        if (check_row(image, l, left, right)) {
            top = l;
        }else{
            top = r;
        }
        //二分最下侧黑色像素坐标
        l = x;
        r = n - 1;
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            if (check_row(image, mid, left, right)) {
                l = mid;
            } else {
                r = mid;
            }
        }

        if (check_row(image, r, left, right)) {
            bottom = r;
        }else{
            bottom = l;
        }
        return (right - left + 1) * (bottom - top + 1);
    }

    //判断列上是否存在黑色像素

    private boolean check_column(char[][] image, int col) {
        for (int i = 0; i < image.length; i++) {
            if (image[i][col] == '1') {
                return true;
            }
        }
        return false;
    }

    //判断行上是否存在黑色像素
    private boolean check_row(char[][] image, int row ,int left ,int right) {
        for (int j = left; j <= right; j++) {
            if (image[row][j] == '1') {
                return true;
            }
        }
        return false;
    }


    public int[] intersection(int[] nums1, int[] nums2) {
        // write your code here
        Arrays.sort(nums1);
        Arrays.sort(nums2);

        int index = 0, index1 = 0, index2 = 0;
        int[] intersection = new int[nums1.length + nums2.length];

        while (index1 < nums1.length && index2 < nums2.length) {
            if (nums1[index1] == nums2[index2]) {
                if (index == 0 || nums1[index1] != intersection[index - 1]) {
                    intersection[index++] = nums1[index1];
                }
                index1++;
                index2++;
            } else if (nums1[index1] < nums2[index2]) {
                index1++;
            } else {
                index2++;
            }
        }

        return Arrays.copyOfRange(intersection, 0, index);
    }

    public int minDepth(TreeNode root) {
        // write your code here
        if (root == null) {
            return 0;
        }

        if (root.left == null && root.right == null) {
            return 1;
        }

        int minDepth = Integer.MAX_VALUE;
        if (root.left != null) {
            minDepth = Math.min(minDepth(root.left), minDepth);
        }

        if (root.right != null) {
            minDepth = Math.min(minDepth(root.right), minDepth);
        }

        return minDepth + 1;
    }

    public int[] getArray(double[] a, int target) {
        if(a == null || a.length == 0) {
            return new int[0];
        }
        int n = a.length;
        double[][] dp = new double[n + 1][target + 1];
        // dp[i][j] means the minimum cost of changing first i numbers and then the sum of them is eqaul to target.
        int[][] nums = new int[n + 1][target + 1];
        // record the int array

        // Initialize the dp array
        for (int i = 0; i < n + 1; i++) {
            for (int j = 0; j < target + 1; j++) {
                dp[i][j] = Double.MAX_VALUE;
            }
        }
        dp[0][0] = 0.0;

        //formula
        //dp[i][j] =  min(dp[i - 1][j - ceil] + ceil - a[i],dp[i - 1][j - floor] + a[i] - floor);
        //nums[i][j] means the integer result of the ith number when the total sum is equal to j;
        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < target + 1; j++) {
                int floor = (int)Math.floor(a[i - 1]);
                int ceil = (int)Math.ceil(a[i - 1]);
                if (j < floor) {
                    continue;
                }
                if (j < ceil) {
                    dp[i][j] = dp[i - 1][j - floor] + a[i - 1] - floor;
                    nums[i][j] = floor;
                    continue;
                }
                //dp[i][j] =  Math.min(dp[i - 1][j - ceil] + ceil - a[i - 1], dp[i - 1][j - floor] + a[i - 1] - floor);
                double costCeil = dp[i - 1][j - ceil] + ceil - a[i - 1];
                double costFloor = dp[i - 1][j - floor] + a[i - 1] - floor;
                if (costFloor <= costCeil) {
                    dp[i][j] = costFloor;
                    nums[i][j] = floor;
                } else {
                    dp[i][j] = costCeil;
                    nums[i][j] = ceil;
                }
            }
        }

        int[] result = new int[n];
        for(int i = n; i > 0; i--) {
            result[i - 1] = nums[i][target];
            target = target - result[i - 1];
        }

        return result;
    }



    public int longestContinuousIncreasingSubsequence2(int[][] matrix) {
        int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int rows, columns;
        // write your code here
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        rows = matrix.length;
        columns = matrix[0].length;
        int[][] outdegrees = new int[rows][columns];
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                for (int[] dir : dirs) {
                    int newRow = i + dir[0], newColumn = j + dir[1];
                    if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] > matrix[i][j]) {
                        ++outdegrees[i][j];
                    }
                }
            }
        }
        Queue<int[]> queue = new LinkedList<int[]>();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                if (outdegrees[i][j] == 0) {
                    queue.offer(new int[]{i, j});
                }
            }
        }
        int ans = 0;
        while (!queue.isEmpty()) {
            ++ans;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cell = queue.poll();
                int row = cell[0], column = cell[1];
                for (int[] dir : dirs) {
                    int newRow = row + dir[0], newColumn = column + dir[1];
                    if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] < matrix[row][column]) {
                        --outdegrees[newRow][newColumn];
                        if (outdegrees[newRow][newColumn] == 0) {
                            queue.offer(new int[]{newRow, newColumn});
                        }
                    }
                }
            }
        }
        return ans;
    }

    public long kDistinctCharacters(String s, int k) {
        // Write your code here
        if (s.length() == 0) {
            return 0;
        }

        long ans = 0;
        int right = 0;
        char c;
        Map<Character, Integer> map = new HashMap<>();

        for (int left = 0; left < s.length(); left++) {
            while (right < s.length() && map.size() < k) {
                c = s.charAt(right);
                map.put(c, map.getOrDefault(c, 0) + 1);
                right++;
            }

            if (map.size() == k) {
                ans += s.length() - right + 1;
            }

            c = s.charAt(left);
            if (map.get(c) > 1) {
                map.put(c, map.get(c) - 1);
            } else {
                map.remove(c);
            }
        }
        return ans;
    }

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    private boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    public int[] searchRange(int[] a, int target) {
        // write your code here
        if (a.length == 0) {
            return new int[]{-1, -1};
        }

        int leftBound = searchRangeHelper(a, target, 0, a.length - 1, false);
        int rightBound = searchRangeHelper(a, target, 0, a.length - 1, true);

        return new int[]{leftBound, rightBound};

    }

    private int searchRangeHelper(int[] a, int target, int left, int right, boolean isUpper) {
        while (left + 1 < right) {
            int mid = (left + right) / 2;

            if (a[mid] > target) {
                right = mid;
            } else if (a[mid] < target) {
                left = mid;
            } else {
                if (isUpper) {
                    left = mid;
                } else {
                    right = mid;
                }
            }
        }

        if (isUpper) {
            if (a[right] == target) {
                return right;
            } else if (a[left] == target) {
                return  left;
            } else {
                return -1;
            }
        } else {
            if (a[left] == target) {
                return left;
            } else if (a[right] == target) {
                return  right;
            } else {
                return -1;
            }
        }
    }

    /*public int findFirstBadVersion(int n) {
        // write your code here
        int left = 1, right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (SVNRepo.isBadVersion(mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }*/

    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        // write your code here
        boolean[][] visited = new boolean[maze.length][maze[0].length];
        int[][] dirs={{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        Queue < int[] > queue = new LinkedList < > ();
        queue.add(start);
        visited[start[0]][start[1]] = true;
        while (!queue.isEmpty()) {
            int[] s = queue.remove();
            if (s[0] == destination[0] && s[1] == destination[1])
                return true;
            for (int[] dir: dirs) {
                int x = s[0] + dir[0];
                int y = s[1] + dir[1];
                while (x >= 0 && y >= 0 && x < maze.length && y < maze[0].length && maze[x][y] == 0) {
                    x += dir[0];
                    y += dir[1];
                }
                if (!visited[x - dir[0]][y - dir[1]]) {
                    queue.add(new int[] {x - dir[0], y - dir[1]});
                    visited[x - dir[0]][y - dir[1]] = true;
                }
            }
        }
        return false;
    }


    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        // write your code here
        List<List<Integer>> ans = new LinkedList<>();
        if (root == null) {
            return ans;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                level.add(node.val);
                TreeNode left = node.left;
                TreeNode right = node.right;
                if (left != null) {
                    queue.offer(left);
                }

                if (right != null) {
                    queue.offer(right);
                }
            }
            ans.add(0, level);
        }

        return ans;
    }

    public int sixDegrees(List<UndirectedGraphNode> graph, UndirectedGraphNode s, UndirectedGraphNode t) {
        // write your code here
        if (graph.size() == 0 || graph == null) {
            return -1;
        }
        if (s == t) {
            return 0;
        }

        Map<UndirectedGraphNode, Integer> visited = new HashMap<>();
        Queue<UndirectedGraphNode> queue = new LinkedList<>();

        queue.offer(s);
        visited.put(s, 0);

        while (!queue.isEmpty()) {
            UndirectedGraphNode node = queue.poll();
            int step = visited.get(node);

            for (UndirectedGraphNode neighbor : node.neighbors) {
                if (visited.containsKey(neighbor)) {
                    continue;
                }

                visited.put(neighbor, step + 1);
                queue.offer(neighbor);
            }
        }

        return visited.getOrDefault(t, -1);
    }

    public int minLength(String s, Set<String> dict) {
        // write your code here
        if (s.length() == 0) {
            return 0;
        }

        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();

        queue.add(s);
        int ans = Integer.MAX_VALUE;

        while (!queue.isEmpty()) {
            String str = queue.poll();
            for (String word : dict) {
                int pos = str.indexOf(word);
                while (pos != -1) {
                    if (str.equals(word)) {
                        return 0;
                    }

                    String temp = str.substring(0, pos) + str.substring(pos + word.length());
                    if (!visited.contains(temp)) {
                        visited.add(temp);
                        ans = Math.min(ans, temp.length());
                        queue.add(temp);
                    }

                    pos = str.indexOf(word, pos + 1);
                }
            }
        }
        return ans;
    }

    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        // write your code here
        if (maze.length == 0) {
            return 0;
        }

        int[][] distance = new int[maze.length][maze[0].length];
        for (int[] row : distance) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }

        distance[start[0]][start[1]] = 0;
        int[][] directions = {{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        Queue<int[]> queue = new LinkedList<>();
        queue.add(start);

        while (!queue.isEmpty()) {
            int[] s = queue.poll();
            for (int[] direction : directions) {
                int x = s[0] + direction[0];
                int y = s[1] + direction[1];
                int count = 0;
                while (x >= 0 && y >= 0 && x < maze.length && y < maze[0].length && maze[x][y] == 0) {
                    x += direction[0];
                    y += direction[1];
                    count++;
                }

                if (distance[s[0]][s[1]] + count < distance[x - direction[0]][y - direction[1]]) {
                    distance[x - direction[0]][y - direction[1]] = distance[s[0]][s[1]] + count;
                    queue.offer(new int[] {x - direction[0], y - direction[1]});
                }
            }
        }

        return distance[destination[0]][destination[1]] == Integer.MAX_VALUE ? -1 : distance[destination[0]][destination[1]];
    }

    public int kthfloorNode(TreeNode root, int k) {
        // Write your code here
        if (root == null) {
            return 0;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        int level = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            if (level == k) {
                return size;
            }

            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            level++;
        }

        return 0;
    }

    public int minMoveStep(int[][] init_state, int[][] final_state) {
        // # write your code here
        String source = matrixToString(init_state);
        String target = matrixToString(final_state);

        Queue<String> queue = new ArrayDeque<>();
        Map<String, Integer> distance = new HashMap<>();

        queue.offer(source);
        distance.put(source, 0);
        while (!queue.isEmpty()) {
            String s = queue.poll();
            if (s.equals(target)) {
                return distance.get(s);
            }
            for (String next : getNextStates(s)) {
                if (distance.containsKey(next)) {
                    continue;
                }
                distance.put(next, distance.get(s) + 1);
                queue.offer(next);
            }
        }

        return -1;
    }

    private List<String> getNextStates(String s) {
        int zeroIndex = s.indexOf("0");
        int x = zeroIndex / 3;
        int y = zeroIndex % 3;
        int[][] DIRECTIONS = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        List<String> nextStates = new ArrayList<>();
        for (int[] dir : DIRECTIONS) {
            int nextX = x + dir[0];
            int nextY = y + dir[1];
            if (!inRange(nextX, nextY)) {
                continue;
            }
            int nextIndex = nextX * 3 + nextY;
            char[] state = s.toCharArray();
            state[zeroIndex] = state[nextIndex];
            state[nextIndex] = '0';
            nextStates.add(new String(state));
        }

        return nextStates;
    }

    private boolean inRange(int x, int y) {
        return x >= 0 && x < 3 && y >= 0 && y < 3;
    }

    private String matrixToString(int[][] matrix) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                builder.append(String.valueOf(matrix[i][j]));
            }
        }
        return builder.toString();
    }

    public boolean isMatchWild(String s, String p) {
        // write your code here
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;
        for (int i = 1; i <= n ; i++) {
            dp[0][i] = dp[0][i - 1] && p.charAt(i - 1) == '*';
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                }

                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                }
            }
        }

        return dp[m][n];
    }

    private Set<String> wordSet;
    private Map<Integer, List<String>> ans = new HashMap<>();
    public List<String> wordBreak(String s, Set<String> wordDict) {
        // write your code here
        wordSet = wordDict;
        wordBreakHelper(s, 0);
        return ans.get(0);
    }

    private void wordBreakHelper(String s, int index) {
        if (!ans.containsKey(index)) {
            if (index == s.length()) {
                ans.put(index, new ArrayList<>());
                return;
            }

            ans.put(index, new ArrayList<>());
            for (int i = index + 1; i <= s.length(); i++) {
                String word = s.substring(index, i);
                if (wordSet.contains(word)) {
                    wordBreakHelper(s, i);
                    if (ans.get(i).isEmpty() && i == s.length()) {
                        ans.get(index).add(word);
                    } else {
                        for (String success : ans.get(i)) {
                            ans.get(index).add(word + " " + success);
                        }
                    }
                }
            }
        }
    }

    public List<ListNode> binaryTreeToLists(TreeNode root) {
        // Write your code here
        List<ListNode> ans = new LinkedList<>();
        if (root == null) {
            return ans;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int size = queue.size();
            ListNode node = null, head = node;

            for (int i = 0; i < size; i++) {
                TreeNode treeNode = queue.poll();
                if (i == 0) {
                    node = new ListNode(treeNode.val);
                    head = node;
                } else {
                    node.next = new ListNode(treeNode.val);
                    node = node.next;
                }

                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }

                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
            }

            ans.add(head);
        }

        return ans;
    }

    public List<List<Integer>> binaryTreePathSum(TreeNode root, int target) {
        // write your code here
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) {
            return ans;
        }

        Deque<Integer> path = new ArrayDeque<>();
        binaryTreePathSumHelper(root, target, path, ans);
        return ans;
    }

    private void binaryTreePathSumHelper(TreeNode node,
                                         int target,
                                         Deque<Integer> path,
                                         List<List<Integer>> ans) {
        if (node == null) {
            return;
        }

        path.addLast(node.val);
        if (node.left == null && node.right == null) {
            if (node.val == target) {
                ans.add(new ArrayList<>(path));
                path.removeLast();
                return;
            }
        }

        binaryTreePathSumHelper(node.left, target - node.val, path, ans);
        binaryTreePathSumHelper(node.right, target - node.val, path, ans);
        path.removeLast();
    }

    public ParentTreeNode lowestCommonAncestorII(ParentTreeNode root, ParentTreeNode A, ParentTreeNode B) {
        // write your code here
        Set<ParentTreeNode> parentSet = new HashSet<>();
        ParentTreeNode pointer = A;
        while (pointer != null) {
            parentSet.add(pointer);
            pointer = pointer.parent;
        }

        pointer = B;
        while (pointer != null) {
            if (parentSet.contains(pointer)) {
                return pointer;
            }
            pointer = pointer.parent;
        }

        return null;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // write your code here
        TreeNode pointer = root;

        while (true) {
            if (p.val < pointer.val && q.val < pointer.val) {
                pointer = pointer.left;
            } else if (p.val > pointer.val && q.val > pointer.val) {
                pointer = pointer.right;
            } else {
                break;
            }
        }

        return pointer;
    }

    public List<List<Integer>> binaryTreePathSum2(TreeNode root, int target) {
        // write your code here
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) {
            return ans;
        }

        pathSum2Helper(root, target, new ArrayList<Integer>(), ans);
        return ans;
    }

    private void pathSum2Helper(TreeNode node, int target, List<Integer> path, List<List<Integer>> ans) {
        if (node == null) {
            return;
        }

        path.add(node.val);
        int sum = 0;
        for (int i = path.size() - 1; i >= 0; i--) {
            sum += path.get(i);

            if (sum == target) {
                ans.add(new ArrayList<Integer>(path.subList(i, path.size())));
            }
        }

        pathSum2Helper(node.left, target, path, ans);
        pathSum2Helper(node.right,target,path,ans);

        path.remove(path.size() - 1);
    }

    Map<Integer, TreeNode> parents = new HashMap<>();
    Set<Integer> lowestVisited = new HashSet<>();
    public TreeNode lowestCommonAncestor3(TreeNode root, TreeNode A, TreeNode B) {
        // write your code here
        if (root == null) {
            return null;
        }
        lowestCommonAncestor3DFS(root);

        while (A != null) {
            lowestVisited.add(A.val);
            A = parents.get(A.val);
        }

        while (B != null) {
            if (lowestVisited.contains(B.val)) {
                return B;
            }
            B = parents.get(B.val);
        }

        return null;
    }

    private void lowestCommonAncestor3DFS(TreeNode node) {
        if (node.left != null) {
            parents.put(node.left.val, node);
            lowestCommonAncestor3DFS(node.left);
        }

        if (node.right != null) {
            parents.put(node.right.val, node);
            lowestCommonAncestor3DFS(node.right);
        }
    }


    public char firstUniqChar(String str) {
        // Write your code here
        Map<Character, Integer> freq = new HashMap<>();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        }

        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (freq.get(c) == 1) {
                return c;
            }
        }

        return ' ';

    }

    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        // write your code here
        TreeNode ans = null;

        while (root != null) {
            if (root.val <= p.val) {
                root = root.right;
            } else {
                ans = root;
                root = root.left;
            }
        }

        return ans;
    }


    public TreeNode inorderPredecessor(TreeNode root, TreeNode p) {
        // write your code here
        TreeNode ans = null;

        while (root != null) {
            if (p.val <= root.val) {
                root = root.left;
            } else {
                if (ans == null || root.val > ans.val) {
                    ans = root;
                }
                root = root.right;
            }
        }
        return ans;
    }

    public int kthLargestElement2(int[] nums, int k) {
        // write your code here
        int heapSize = nums.length;
        buildMaxHeap(nums, heapSize);
        for (int i = nums.length - 1; i >= nums.length - k + 1; i--) {
            swap(nums, 0, i);
            heapSize--;
            maxHeapify(nums, 0, heapSize);
        }

        return nums[0];
    }

    private void buildMaxHeap(int[] nums, int heapSize) {
        for (int i = heapSize / 2; i >= 0; i--) {
            maxHeapify(nums, i, heapSize);
        }
    }

    private void maxHeapify(int[] nums, int i, int heapSize) {
        int l = i * 2 + 1;
        int r = i * 2 + 2;
        int largest = i;

        if (l < heapSize && nums[l] > nums[largest]) {
            largest = l;
        }

        if (r < heapSize && nums[r] > nums[largest]) {
            largest = r;
        }

        if (largest != i) {
            swap(nums, i, largest);
            maxHeapify(nums, largest, heapSize);
        }

    }

    public ListNode detectCycle(ListNode head) {
        // write your code here
        if (head == null) {
            return null;
        }

        ListNode slow = head, fast = head;
        while (fast != null) {
            slow = slow.next;
            if (fast.next != null) {
                fast = fast.next.next;
            } else {
                return null;
            }

            if (fast == slow) {
                ListNode pointer = head;
                while (pointer != slow) {
                    pointer = pointer.next;
                    slow = slow.next;
                }
                return pointer;
            }
        }

        return null;
    }

    public double mincostToHireWorkers(int[] quality, int[] wage, int k) {
        // Write your code here
        int N = quality.length;
        Worker[] workers = new Worker[N];
        for (int i = 0; i < N; i++) {
            workers[i] = new Worker(quality[i], wage[i]);
        }

        Arrays.sort(workers);

        double ans = 1e9;
        int sum = 0;
        PriorityQueue<Integer> pool = new PriorityQueue<>();
        for (Worker worker: workers) {
            pool.offer(-worker.quality);
            sum += worker.quality;
            if (pool.size() > k) {
                sum += pool.poll();
            }
            if (pool.size() == k) {
                ans = Math.min(ans, sum * worker.ratio());
            }
        }

        return ans;
    }


    public int totalNQueens(int n) {
        // write your code here
        Set<Integer> columns = new HashSet<>();
        Set<Integer> diagonals1 = new HashSet<>();
        Set<Integer> diagonals2 = new HashSet<>();
        return totalNQueensBacktrack(n, 0, columns, diagonals1, diagonals2);
    }

    private int totalNQueensBacktrack(int n, int row, Set<Integer> columns, Set<Integer> diagonals1, Set<Integer> diagonals2) {
        if (row == n) {
            return 1;
        } else {
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (columns.contains(i)) {
                    continue;
                }
                int diagonal1 = row - i;
                if (diagonals1.contains(diagonal1)) {
                    continue;
                }
                int diagonal2 = row + i;
                if (diagonals2.contains(diagonal2)) {
                    continue;
                }
                columns.add(i);
                diagonals1.add(diagonal1);
                diagonals2.add(diagonal2);

                count += totalNQueensBacktrack(n, row + 1, columns, diagonals1, diagonals2);

                columns.remove(i);
                diagonals1.remove(diagonal1);
                diagonals2.remove(diagonal2);
            }

            return count;
        }
    }

    public List<List<Integer>> combine(int n, int k) {
        // write your code here
        List<Integer> combination = new ArrayList<>();
        List<List<Integer>> ans = new ArrayList<>();

        combineHelper(1, n, k, combination, ans);

        return ans;
    }

    private void combineHelper(int cur, int n, int k, List<Integer> combination, List<List<Integer>> ans) {
        if (combination.size() + (n - cur + 1) < k) {
            return;
        }

        if (combination.size() == k) {
            ans.add(new ArrayList<>(combination));
            return;
        }

        combination.add(cur);
        combineHelper(cur + 1, n, k, combination, ans);
        combination.remove(combination.size() - 1);
        combineHelper(cur + 1, n, k, combination, ans);
    }

    public List<List<Integer>> combinationSum2(int[] num, int target) {
        // write your code here
        List<List<Integer>> ans = new ArrayList<>();
        if (num == null || num.length == 0) {
            return ans;
        }
        List<Integer> path = new ArrayList<>();
        Arrays.sort(num);
        sum2Helper(num, ans, path, target, 0, false);
        return ans;
    }

    private void sum2Helper(int[] num,
                            List<List<Integer>> ans,
                            List<Integer> path,
                            int target,
                            int index,
                            boolean selected) {
        if (target == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }

        if (index >= num.length) {
            return;
        }

        if (num[index] <= target &&
                (index < 1 || selected || num[index] != num[index - 1])) {
            path.add(num[index]);
            sum2Helper(num, ans, path, target - num[index], index + 1, true);
            path.remove(path.size() - 1);
        }

        sum2Helper(num, ans, path, target, index + 1, false);
    }

    static final int SEG_COUNT = 4;
    List<String> ipAns = new ArrayList<>();
    int[] segments = new int[SEG_COUNT];
    public List<String> restoreIpAddresses(String s) {
        // write your code here
        restoreIpHelper(s, 0, 0);
        return ipAns;
    }

    private void restoreIpHelper(String s, int segId, int segStart) {
        if (segId == SEG_COUNT) {
            if (segStart == s.length()) {
                StringBuffer ipAddr = new StringBuffer();
                for (int i = 0; i < SEG_COUNT; i++) {
                    ipAddr.append(segments[i]);
                    if (i != SEG_COUNT - 1) {
                        ipAddr.append('.');
                    }
                }
                ipAns.add(ipAddr.toString());
            }
            return;
        }

        if (segStart == s.length()) {
            return;
        }

        if (s.charAt(segStart) == '0') {
            segments[segId] = 0;
            restoreIpHelper(s, segId + 1, segStart + 1);
        }

        int addr = 0;
        for (int i = segStart; i < s.length(); i++) {
            addr = addr * 10 + (s.charAt(i) - '0');
            if (addr > 0 && addr <= 0xFF) {
                segments[segId] = addr;
                restoreIpHelper(s, segId + 1, i + 1);
            } else {
                break;
            }
        }
    }


    private int theMissing = -1;
    public int findMissing2(int n, String s) {
        // write your code here
        boolean[] isFound = new boolean[n + 1];
        missingDFS(n, s, 0, isFound);
        return theMissing;
    }

    private void missingDFS(int n, String s, int start, boolean[] isFound) {
        if (theMissing != -1) {
            return;
        }

        if (start == s.length()) {
            for (int i = 1; i <= n; i++) {
                if (!isFound[i]) {
                    theMissing = i;
                    return;
                }
            }
        }

        if (s.charAt(start) == '0') {
            return;
        }

        for (int i = 1; i <= 2 && start + i <= s.length() ; i++) {
            int num = Integer.parseInt(s.substring(start, start + i));
            if (num > 0 && num <= n && !isFound[num]) {
                isFound[num] = true;
                missingDFS(n, s, start + i, isFound);
                isFound[num] = false;
            }
        }

        return;
    }

    public List<List<String>> splitString(String s) {
        // write your code here
        List<List<String>> ans = new ArrayList<>();
        splitStringDFS(0, s, new ArrayList<>(), ans);
        return ans;
    }

    private void splitStringDFS(int index, String s, List<String> path, List<List<String>> ans) {
        if (index == s.length()) {
            ans.add(new ArrayList<>(path));
            return;
        }

        for (int i = 1; i <= 2 && index + i <= s.length() ; i++) {
            path.add((s.substring(index, index + i)));
            splitStringDFS(index + i, s, path, ans);
            path.remove(path.size() - 1);
        }
    }

    private List<String> parentheseAns = new ArrayList<>();
    public List<String> removeInvalidParentheses(String s) {
        // Write your code here
        int left = 0, right = 0;

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                left++;
            } else if (s.charAt(i) == ')') {
                if (left == 0) {
                    right++;
                } else {
                    left--;
                }
            }
        }

        removeDFS(s, 0, left, right);

        return parentheseAns;
    }

    private void removeDFS(String s, int index, int left, int right) {
        if (left == 0 && right == 0) {
            if (isValid(s)) {
                parentheseAns.add(s);
            }
            return;
        }

        for (int i = index; i < s.length(); i++) {
            if (i != index && s.charAt(i) == s.charAt(i - 1)) {
                continue;
            }

            if (left + right > s.length() - i) {
                return;
            }

            if (left > 0 && s.charAt(i) == '(') {
                removeDFS(s.substring(0, i) + s.substring(i + 1), i, left - 1, right);
            }

            if (right > 0 && s.charAt(i) == ')') {
                removeDFS(s.substring(0, i) + s.substring(i + 1), i, left, right - 1);
            }
        }
    }

    private boolean isValid(String str) {
        int count = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '(') {
                count++;
            } else if (str.charAt(i) == ')') {
                count--;
                if (count < 0) {
                    return false;
                }
            }
        }

        return count == 0;
    }

    public void solveSudoku(int[][] board) {
        // write your code here
        sudokuHelper(board, 0, 0);
    }

    private boolean sudokuHelper(int[][] board, int i, int j) {
        int m = 9, n = 9;

        if (j == n) {
            return sudokuHelper(board, i + 1, 0);
        }

        if (i == m) {
            return true;
        }

        if (board[i][j] != 0) {
            return sudokuHelper(board, i, j + 1);
        }

        for (int num = 1; num <= 9; num++) {
            if (!sudokuValid(board, i, j, num)) {
                continue;
            }

            board[i][j] = num;
            if (sudokuHelper(board, i, j + 1)) {
                return true;
            }

            board[i][j] = 0;
        }

        return false;
    }

    private boolean sudokuValid(int[][] board, int row, int column, int num) {
        for (int i = 0; i < 9; i++) {
            if (board[row][i] == num) {
                return false;
            }

            if (board[i][column] == num) {
                return false;
            }

            if (board[(row / 3) * 3 + i / 3][(column / 3) * 3 + i % 3] == num) {
                return false;
            }
        }

        return true;
    }

    public boolean wordPatternMatch(String pattern, String str) {
        // write your code here
        Map<Character, String> map = new HashMap<>();
        Set<String> used = new HashSet<>();
        return match(pattern, str, map, used);
    }

    private boolean match(String pattern,
                          String str,
                          Map<Character, String> map,
                          Set<String> used) {
        if (pattern.length() == 0) {
            return str.length() == 0;
        }

        Character c = pattern.charAt(0);

        if (map.containsKey(c)) {
            String word = map.get(c);
            if (!str.startsWith(word)) {
                return false;
            }

            return match(pattern.substring(1),str.substring(map.get(c).length()), map, used);
        }

        for (int i = 0; i < str.length(); i++) {
            String word = str.substring(0, i + 1);
            if (used.contains(word)) {
                continue;
            }

            map.put(c, word);
            used.add(word);

            if (match(pattern.substring(1),str.substring(i + 1), map, used)) {
                return true;
            }

            map.remove(c);
            used.remove(word);
        }
        return false;
    }

    public int minDistance(String word1, String word2) {
        // write your code here
        int n = word1.length();
        int m = word2.length();

        if (n * m == 0) {
            return n + m;
        }

        int[][] dp = new int[n + 1][m + 1];

        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = i;
        }

        for (int i = 0; i < m + 1; i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < m + 1; j++) {
                int left = dp[i - 1][j] + 1;
                int down = dp[i][j -1] + 1;
                int leftDown = dp[i - 1][j - 1];
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    leftDown += 1;
                }

                dp[i][j] = Math.min(left, Math.min(down, leftDown));
            }
        }

        return dp[n][m];
    }

    public int maxProfit(int[] prices) {
        // write your code here
        if (prices.length == 0) {
            return 0;
        }

        int n = prices.length;
        int buy1 = -prices[0], sell1 = 0;
        int buy2 = -prices[0], sell2 = 0;

        for (int i = 0; i < n; i++) {
            buy1 = Math.max(buy1, -prices[i]);
            sell1 = Math.max(sell1, buy1 + prices[i]);
            buy2 = Math.max(buy2, sell1 - prices[i]);
            sell2 = Math.max(sell2, buy2 + prices[i]);
        }

        return sell2;
    }

    public int stoneGame(int[] A) {
        int size = A.length;
        if (A == null || size == 0) {
            return 0;
        }

        int[][] dp = new int[size][size];
        int[] sum_a = new int[size + 1];
        //前缀和
        for (int i = 0; i < size; i++) {
            sum_a[i + 1] = sum_a[i] + A[i];
        }
        // 长度从2开始即可，因为长度为1的时候结果是0，dp初始化的时候默认就是0，没必要赋值
        for (int len = 2; len <= size; len++) {
            // i枚举的是正在枚举的区间的左端点
            for (int i = 0; i + len - 1 < size; i++) {
                // 正在枚举的区间左端点是i，右端点是i + size - 1
                int l = i, r = i + len - 1;
                // 在求最小的时候，需要初始化成一个很大的数，然后不断更新
                dp[l][r] = Integer.MAX_VALUE;
                for (int j = l; j < r; j++) {
                    //递推式
                    dp[l][r] = Math.min(dp[l][r], dp[l][j] + dp[j + 1][r] + sum_a[r + 1] - sum_a[l]);
                }
            }
        }

        return dp[0][size-1];
    }

    public int minCost(int[][] costs) {
        // write your code here
        int n = costs.length;

        if (n == 0) {
            return 0;
        }

        int[][] dp = new int[2][3];
        for (int i = 0; i < 3; i++) {
            dp[0][i] = costs[0][i];
        }

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < 3; j++) {
                dp[i & 1][j] = Integer.MAX_VALUE;
                for (int k = 0; k < 3; k++) {
                    if (k != j) {
                        dp[i & 1][j] = Math.min(dp[i & 1][j], dp[i & 1 ^ 1][k] + costs[i][j]);
                    }
                }
            }
        }

        return Math.min(dp[n & 1 ^ 1][0], Math.min(dp[n & 1 ^ 1][1], dp[n & 1 ^ 1][2]));
    }

    public int maxValue(String str) {
        // write your code here
        int n = str.length();
        int[][] dp = new int[n][n];

        for (int i = 0; i < n; i++) {
            dp[i][i] = (int) str.charAt(i) - (int)'0';
        }

        for (int l = 2; l <= n; l++) {
            for (int i = 0; i < n - l + 1; i++) {
                int j = i + l - 1;
                for (int k = i; k < j; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][k] + dp[k + 1][j]);
                    dp[i][j] = Math.max(dp[i][j], dp[i][k] * dp[k + 1][j]);
                }
            }
        }

        return dp[0][n - 1];
    }

    public int minimumDeleteSum(String s1, String s2) {
        // Write your code here
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];

        for (int i = s1.length() - 1; i >= 0; i--) {
            dp[i][s2.length()] = dp[i + 1][s2.length()] + s1.codePointAt(i);
        }

        for (int i = s2.length() - 1; i >= 0; i--) {
            dp[s1.length()][i] = dp[s1.length()][i + 1] + s2.codePointAt(i);
        }

        for (int i = s1.length() - 1; i >= 0; i--) {
            for (int j = s2.length() - 1; j >= 0; j--) {
                if (s1.charAt(i) == s2.charAt(j)) {
                    dp[i][j] = dp[i + 1][j + 1];
                } else {
                    dp[i][j] = Math.min(dp[i+1][j] + s1.codePointAt(i),
                                        dp[i][j+1] + s2.codePointAt(j));
                }
            }
        }
        return dp[0][0];
    }

    public boolean canCross(int[] stones) {
        // write your code here
        if(stones == null || stones.length == 0){
            return false;
        }
        if(stones.length == 1 || stones.length == 2){
            return true;
        }
        int n = stones.length;
        boolean[][] dp = new boolean[n][n - 1];
        dp[1][1] = true;
        for(int i = 2; i < n; i++){
            for(int j = i - 1; j >= 0; j--){
                int distance = stones[i] - stones[j];
                if(distance > j + 1){
                    break;
                }
                if(dp[j][distance - 1] || dp[j][distance] || dp[j][distance + 1]){
                    dp[i][distance] = true;
                }
            }
        }
        for(int i = 0; i < n - 1; i++){
            if(dp[n - 1][i]){
                return true;
            }
        }
        return false;
    }

    public boolean isValidParentheses(String s) {
        // write your code here
        if (s.length() % 2 != 0) {
            return false;
        }

        Map<Character, Character> map = new HashMap<>();

        map.put(')', '(');
        map.put(']', '[');
        map.put('}', '{');

        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (map.containsKey(c)) {
                if (stack.isEmpty() || stack.peek() != map.get(c)) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(c);
            }
        }

        return stack.isEmpty();
    }

    int diameter;
    public int diameterOfBinaryTree(TreeNode root) {
        // write your code here
        diameter = 1;
        diameterHelper(root);
        return diameter - 1;
    }

    private int diameterHelper(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int left = diameterHelper(node.left);
        int right = diameterHelper(node.right);

        diameter = Math.max(diameter, left + right + 1);
        return Math.max(left, right) + 1;
    }

    public void rerange(int[] a) {
        // write your code here
        if(a == null || a.length < 3) {
            return;
        }

        int n = a.length;
        int countPositive = 0;//count the number of positive numbers

        // store the positive numbers index.
        int positiveIndex = 0;
        int pos = 1;
        int neg = 0;
        for (int i = 0;i < n;i++) {
            if (a[i] > 0) {
                // Put all the positive numbers at in the left part.
                swap(a,positiveIndex++,i);
                countPositive++;
            }
        }

        if (countPositive > n/2) {
            // If positive numbers are more than negative numbers,
            // Put the positive numbers at first.
            pos = 0;
            neg = 1;
            // Reverse the array.

            int left = 0;
            int right = n-1;
            while (left < right) {
                swap(a, left, right);
                left++;
                right--;
            }
        }

        while(pos < n && neg < n) {
            while(pos < n && a[pos] > 0)
                pos +=2;
            while(neg < n && a[neg] < 0)
                neg +=2;
            if(neg >= n || pos >= n)
                break;
            swap(a, pos, neg);
        }
    }

    public int triangleCount(int[] s) {
        // write your code here
        int n = s.length;
        Arrays.sort(s);
        int ans = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int left = j + 1, right = n - 1, k = j;

                while (left <= right) {
                    int mid = (left + right) / 2;
                    if (s[mid] < s[i] + s[j]) {
                        k = mid;
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
                ans += k - j;
            }
        }

        return ans;
    }

    public int twoSum2(int[] nums, int target) {
        // write your code here
        int n = nums.length;
        Arrays.sort(nums);

        int ans = 0, left = 0, right = n - 1;

        while (left < right) {
            if (nums[left] + nums[right] <= target) {
                left++;
            } else {
                ans += right - left;
                right--;
            }
        }

        return ans;
    }

    public int kthSmallest(int k, int[] nums) {
        // write your code here
        int n = nums.length;
        return kthHelper(nums, 0, n - 1, k - 1);
    }

    private int kthHelper(int[] nums, int start, int end, int k) {
        int left = start, right = end;
        int pivot = nums[start];

        while (left <= right) {
            while (left <= right && nums[left] < pivot) {
                left++;
            }

            while (left <= right && nums[right] > pivot) {
                right--;
            }

            if (left <= right) {
                swap(nums, left, right);
                left++;
                right--;
            }
        }

        if (k <= right) {
            return kthHelper(nums, start, right, k);
        }

        if (k >= left){
            return kthHelper(nums, left, end, k);
        }

        return nums[k];
    }

    public int twoSumClosest(int[] nums, int target) {
        // write your code here
        if (nums.length == 0) {
            return -1;
        }

        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        int ans = Integer.MAX_VALUE;

        while (left < right) {
            if (nums[right] + nums[left] == target) {
                return 0;
            } else if (nums[left] + nums[right] < target) {
                ans = Math.min(ans, Math.abs(target - nums[left] - nums[right]));
                left++;
            } else {
                ans = Math.min(ans, Math.abs(target - nums[left] - nums[right]));
                right--;
            }
        }

        return ans;
    }

    public int[] twoSumII(int[] nums, int target) {
        // write your code here
        if (nums.length == 0) {
            return new int[]{-1, -1};
        }

        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] == target) {
                return new int[]{left + 1, right + 1};
            } else if (nums[left] + nums[right] <  target) {
                left++;
            } else {
                right--;
            }
        }

        return new int[]{-1, -1};
    }

    public int twoSum5II(int[] nums, int target) {
        // write your code here
        if (nums.length == 0) {
            return 0;
        }

        Arrays.sort(nums);

        int left = 0, right = nums.length - 1;
        int count = 0;

        while (left <= nums.length - 1) {
            while (right >= 0) {
                if (nums[left] + nums[right] > target) {
                    right--;
                } else {
                    break;
                }
            }

            if (left < right) {
                count += right - left;
            } else {
                break;
            }

            left++;
        }

        return count;
    }

    public int search(int[] a, int target) {
        // write your code here
        int n = a.length;
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return a[0] == target ? 0 : -1;
        }

        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (a[mid] == target) {
                return mid;
            }

            if (a[0] <= a[mid]) {
                if (a[0] <= target && target < a[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (a[mid] < target && target <= a[n - 1]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        return -1;
    }

//    public int searchBigSortedArray(ArrayReader reader, int target) {
//        // write your code here
//        int k = 1;
//        while (reader.get(k - 1) < target) {
//            k = k * 2;
//        }
//
//        int start = 0, end = k - 1;
//        while (start + 1 < end) {
//            int mid = start + (end - start) / 2;
//            if (reader.get(mid) < target) {
//                start = mid;
//            } else {
//                end = mid;
//            }
//        }
//
//        if (reader.get(start) == target) {
//            return start;
//        }
//
//        if (reader.get(end) == target) {
//            return end;
//        }
//
//        return -1;
//    }

    public int strStr2(String source, String target) {
        // write your code here
        if (source == null || target == null) {
            return -1;
        }

        int m = target.length();
        if (m == 0) {
            return 0;
        }

        int power = 1;
        int BASE = 100007;
        for (int i = 0; i < m; i++) {
            power = (power * 31) % BASE;
        }

        int targetCode = 0;
        for (int i = 0; i < m; i++) {
            targetCode = (targetCode * 31 + target.charAt(i)) % BASE;
        }

        int sourceCode = 0;
        for (int i = 0; i < source.length(); i++) {
            sourceCode = (sourceCode * 31 + source.charAt(i)) % BASE;

            if (m - 1 >= i) {
                continue;
            }

            sourceCode = (sourceCode - power * source.charAt(i - m)) % BASE;

            if (sourceCode < 0) {
                sourceCode += BASE;
            }

            if (sourceCode == targetCode) {
                return i - m + 1;
            }
        }

        return -1;
    }

    public boolean validTree(int n, int[][] edges) {
        // write your code here
        if (n == 0) {
            return false;
        }

        if (edges.length != n - 1) {
            return false;
        }

        Map<Integer, Set<Integer>> graph = initializeGraph(n, edges);
        Set<Integer> visited = new HashSet<>();
        //validTreeBFS(0, graph, visited);
        validTreeDFS(0, graph, visited);

        return visited.size() == n;
    }

    private void validTreeBFS(int n, Map<Integer, Set<Integer>> graph, Set<Integer> visited) {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(n);
        visited.add(n);

        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (Integer neighbor : graph.get(node)) {
                if (visited.contains(neighbor)) {
                    continue;
                }
                queue.offer(neighbor);
                visited.add(neighbor);
            }
        }
    }

    private void validTreeDFS(int n, Map<Integer, Set<Integer>> graph, Set<Integer> visited) {
        visited.add(n);
        for (Integer neighbor : graph.get(n)) {
            if (visited.contains(neighbor)) {
                continue;
            }
            validTreeDFS(neighbor, graph, visited);
        }
    }

    private Map<Integer, Set<Integer>> initializeGraph(int n, int[][] edges) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graph.put(i, new HashSet<>());
        }

        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }

        return graph;
    }

    public List<List<Integer>> connectedSet(List<UndirectedGraphNode> nodes) {
        // write your code here
        List<List<Integer>> ans = new ArrayList<>();
        if (nodes == null || nodes.size() == 0) {
            return ans;
        }

        Set<UndirectedGraphNode> visited = new HashSet<>();

        for (UndirectedGraphNode node : nodes) {
            if (!visited.contains(node)) {
                connectedSetBFS(node, visited, ans);
            }
        }
        return ans;
    }

    private void connectedSetBFS(UndirectedGraphNode start, Set<UndirectedGraphNode> visited, List<List<Integer>> ans) {
        Queue<UndirectedGraphNode> queue = new LinkedList<>();
        List<Integer> group = new ArrayList<>();
        queue.offer(start);
        visited.add(start);
        group.add(start.label);

        while (!queue.isEmpty()) {
            UndirectedGraphNode node = queue.poll();
            for (UndirectedGraphNode neighbor : node.neighbors) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                    group.add(neighbor.label);
                }
            }
        }

        Collections.sort(group);
        ans.add(group);
    }

    public int zombie(int[][] grid) {
        // write your code here
        int n = grid.length;
        int m = grid[0].length;

        Queue<Point> queue = new LinkedList<>();
        int human = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
               if (grid[i][j] == 1) {
                   queue.add(new Point(i, j));
               } else if (grid[i][j] == 0) {
                   human += 1;
               }
            }
        }

        final int[][] DIRECTIONS = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        int count = 0;
        while (!queue.isEmpty()) {
            int zombieIncreased = 0;
            int newZombie = queue.size();
            for (int i = 0; i < newZombie; i++) {
                Point point = queue.poll();
                for (int[] direction : DIRECTIONS) {
                    int newX = direction[0] + point.x;
                    int newY = direction[1] + point.y;
                    if (boundaryCheck(newX, newY, n, m, grid)) {
                        queue.add(new Point(newX, newY));
                        grid[newX][newY] = 1;
                        zombieIncreased++;
                    }
                }
            }

            if (zombieIncreased == 0) {
                break;
            }

            human -= zombieIncreased;
            count++;

            if (human == 0) {
                break;
            }
        }

        if (human > 0) {
            return -1;
        }

        return count;
    }

    private boolean boundaryCheck(int x, int y, int n, int m, int[][] grid) {
        if (x < 0 || x >= n || y < 0 || y >= m) {
            return false;
        }

        return grid[x][y] == 0;
    }

    public boolean sequenceReconstruction(int[] org, int[][] seqs) {
        // write your code here
        Map<Integer, Set<Integer>> graph = sequenceGraph(seqs);
        Map<Integer, Integer> inDegree = getIndegree(seqs, graph);

        List<Integer> zeroInDegree = new ArrayList<>();

        for (int i : inDegree.keySet()) {
            if (inDegree.get(i) == 0) {
                zeroInDegree.add(i);
            }
        }

        Queue<Integer> queue = new LinkedList<>();
        List<Integer> result = new ArrayList<>();

        for (int i : zeroInDegree) {
            queue.offer(i);
        }

        while (!queue.isEmpty()) {
            int size = queue.size();

            if (size > 1) {
                return false;
            }

            int i = queue.poll();
            result.add(i);

            if (!graph.containsKey(i)) {
                continue;
            }

            for (int next : graph.get(i)) {
                inDegree.put(next, inDegree.get(next) - 1);
                if (inDegree.get(next) == 0) {
                    queue.offer(next);
                }
            }
        }

        if (result.size() != org.length) {
            return false;
        }

        for (int i = 0; i < org.length; i++) {
            if (org[i] != result.get(i)) {
                return false;
            }
        }

        return true;
    }

    private Map<Integer, Integer> getIndegree(int[][] seqs, Map<Integer, Set<Integer>> graph) {
        Map<Integer, Integer> inDegree = new HashMap<>();

        for (int i = 0; i < seqs.length; i++) {
            for (int j = 0; j < seqs[i].length; j++) {
                inDegree.put(seqs[i][j], 0);
            }
        }

        for (int i : graph.keySet()) {
            for (int next : graph.get(i)) {
                inDegree.put(next, inDegree.get(next) + 1);
            }
        }

        return inDegree;
    }

    private Map<Integer, Set<Integer>> sequenceGraph(int[][] seqs) {
        Map<Integer, Set<Integer>> graph = new HashMap<>();

        for (int i = 0; i < seqs.length; i++) {
            for (int j = 0; j < seqs[i].length - 1; j++) {
                if (!graph.containsKey(seqs[i][j])) {
                    Set<Integer> set = new HashSet<>();
                    set.add(seqs[i][j + 1]);
                    graph.put(seqs[i][j], set);
                } else {
                    graph.get(seqs[i][j]).add(seqs[i][j + 1]);
                }
            }
        }
        return graph;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // write your code here
        List<ArrayList<Integer>> graph = new ArrayList<>();
        int[] inDegree = new int[numCourses];

        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }

        for (int[] prereq : prerequisites) {
            graph.get(prereq[1]).add(prereq[0]);
            inDegree[prereq[0]]++;
        }

        int numChoose = 0;
        Queue<Integer> queue = new LinkedList<>();

        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] == 0) {
                queue.add(i);
            }
        }

        while (!queue.isEmpty()) {
            int index = queue.poll();
            numChoose++;

            for (int i = 0; i < graph.get(index).size(); i++) {
                int nextIndex = graph.get(index).get(i);
                inDegree[nextIndex]--;
                if (inDegree[nextIndex] == 0) {
                    queue.add(nextIndex);
                }
            }
        }

        return numChoose == numCourses;
    }


    public int[] findOrder(int numCourses, int[][] prerequisites) {
        // write your code here
        List<ArrayList<Integer>> graph = new ArrayList<>();
        int[] inDegree = new int[numCourses];

        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }

        for (int[] prereq : prerequisites) {
            graph.get(prereq[1]).add(prereq[0]);
            inDegree[prereq[0]]++;
        }

        int numChoose = 0;
        Queue<Integer> queue = new LinkedList<>();
        int[] topoOrder = new int[numCourses];

        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] == 0) {
                queue.add(i);
            }
        }

        while (!queue.isEmpty()) {
            int index = queue.poll();
            topoOrder[numChoose] = index;
            numChoose++;
            for (int i = 0; i < graph.get(index).size(); i++) {
                int nextIndex = graph.get(index).get(i);
                inDegree[nextIndex]--;
                if (inDegree[nextIndex] == 0) {
                    queue.add(nextIndex);
                }
            }
        }

        if (numChoose == numCourses) {
            return topoOrder;
        }

        return new int[0];
    }

    public UndirectedGraphNode searchNode(ArrayList<UndirectedGraphNode> graph,
                                          Map<UndirectedGraphNode, Integer> values,
                                          UndirectedGraphNode node,
                                          int target) {
        // write your code here
        if (graph == null || values == null || node == null) {
            return null;
        }

        Queue<UndirectedGraphNode> queue = new LinkedList<>();
        Set<UndirectedGraphNode> visited = new HashSet<>();

        queue.add(node);
        visited.add(node);

        while (!queue.isEmpty()) {
            UndirectedGraphNode cur = queue.poll();
            if (values.get(cur) == target) {
                return cur;
            }

            for (UndirectedGraphNode neighbor : cur.neighbors) {
                if (!visited.contains(neighbor)) {
                    queue.add(neighbor);
                    visited.add(neighbor);
                }
            }
        }

        return null;
    }

    public TreeNode insertNode(TreeNode root, TreeNode node) {
        // write your code here
        if (root == null) {
            return node;
        }

        TreeNode cur = root;
        while (true) {
            if (node.val < cur.val) {
                if (cur.left == null) {
                    cur.left = node;
                    break;
                } else {
                    cur = cur.left;
                }
            } else {
                if (cur.right == null) {
                    cur.right = node;
                    break;
                } else {
                    cur = cur.right;
                }
            }
        }

        return root;
    }

    private TreeNode maxAve = null;
    private AnswerSub maxAns = null;
    public TreeNode findSubtree2(TreeNode root) {
        // write your code here
        findSubtree2Helper(root);
        return maxAve;
    }

    private AnswerSub findSubtree2Helper(TreeNode node) {
        if (node == null) {
            return new AnswerSub(0, 0);
        }

        AnswerSub left = findSubtree2Helper(node.left);
        AnswerSub right = findSubtree2Helper(node.right);

        AnswerSub res = new AnswerSub(left.sum + right.sum + node.val, left.count + right.count + 1);
        if (maxAve == null || (double) maxAns.sum / maxAns.count < (double) res.sum / res.count) {
            maxAve = node;
            maxAns = res;
        }

        return res;
    }

    public List<Integer> searchRange(TreeNode root, int k1, int k2) {
        // write your code here
        List<Integer> ans = new ArrayList<>();
        searchHelper(ans, root, k1, k2);
        return ans;
    }

    private void searchHelper(List<Integer> ans, TreeNode node, int k1, int k2) {
        if (node == null) {
            return;
        }

        if (node.val > k2) {
            searchHelper(ans, node.left, k1, k2);
        }

        if (node.val >= k1 && node.val <= k2) {
            searchHelper(ans, node.left, k1, k2);
            ans.add(node.val);
            searchHelper(ans, node.right, k1, k2);
        }

        if (node.val < k1) {
            searchHelper(ans, node.right, k1, k2);
        }
    }

    public boolean isValidBST(TreeNode root) {
        // write your code here
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean isValidBST(TreeNode node, long lower, long upper) {
        if (node == null) {
            return true;
        }

        if (node.val <= lower || node.val >= upper) {
            return false;
        }

        return isValidBST(node.left, lower, node.val) && isValidBST(node.right, node.val, upper);
    }

    public int longestConsecutive(int[] num) {
        // write your code here
        Set<Integer> set = new HashSet<>();

        for (int i = 0; i < num.length; i++) {
            set.add(num[i]);
        }

        int longestStreak = 0;

        for (int x : num) {
            if (!set.contains(x - 1)) {
                int cur = x;
                int currentStreak = 1;

                while (set.contains(++cur)) {
                    currentStreak++;
                }

                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }

    public String stringReplace(String[] a, String[] b, String s) {
        // Write your code here
        int n = a.length;
        int len = s.length();
        //hash函数的基底
        long seed = 31L;
        //hash函数的模数
        long mod = 1000000007L;
        //a数组里每个字符串的hash值
        List<Long> aHash = new ArrayList<Long>();
        //s串前缀hash值
        List<Long> sHash = new ArrayList<Long>();
        //基底值
        List<Long> base = new ArrayList<Long>();
        // 答案，为了避免MLE 不能用string
        StringBuilder ans = new StringBuilder(s);
        for(int i = 0; i < n; i++) {
            long tmp = 0L;
            //计算a数组中第i串的hash值
            for(int j = 0; j < a[i].length(); j++) {
                tmp = tmp * seed + ((int)(a[i].charAt(j)) - (int)('a'));
                tmp = tmp % mod;
            }
            aHash.add(tmp);
        }
        long sTmp = 0L, baseTmp = 1L;
        sHash.add(sTmp);
        base.add(baseTmp);
        for(int i = 0; i < len; i++) {
            //计算s串的前缀hash值
            sTmp = sTmp * seed + (int)(s.charAt(i)) - (int)('a');
            sTmp %= mod;
            sHash.add(sTmp);



            //计算基底
            baseTmp = baseTmp * seed;
            baseTmp %= mod;
            base.add(baseTmp);

        }
        int i = 0;
        while(i < len) {
            int mx = 0, idx = -1;
            //枚举和a数组中哪个字符串匹配，且寻找最长的那个
            for(int j = 0; j < a.length; j++) {
                int aLen = a[j].length();
                if(i + aLen > len)continue;
                long A = aHash.get(j);
                //hash计算s的子串的hash值
                long S = sHash.get(aLen + i) - base.get(aLen) * sHash.get(i) % mod;
                A = A % mod;
                S = (S % mod + mod) % mod;
                if(A == S && aLen > mx) {
                    mx = aLen;
                    idx = j;
                }
            }
            if(idx != -1) {
                //有匹配成功的，用b数组的字符串替换掉
                ans.replace(i, i + mx, b[idx]);
                i = i + mx;
            } else {
                //保留原s串的字符
                i = i + 1;
            }
        }
        return ans.toString();
    }


    public int[] deltaX = {0, 1, 0, -1};
    public int[] deltaY = {1, 0, -1, 0};
    public int shortestDistance(int[][] grid) {
        // write your code here
        if (grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int n = grid.length, m = grid[0].length;
        int minDist = Integer.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == GridType.EMPTY) {
                    Map<Integer, Integer> distance = shortestHelper(grid, i ,j);
                    minDist = Math.min(minDist, getDistanceSum(distance, grid));
                }
            }
        }

        return minDist != Integer.MAX_VALUE ? minDist : -1;
    }

    private Map<Integer, Integer> shortestHelper(int[][] grid, int i, int j) {
        int n = grid.length, m = grid[0].length;

        Map<Integer, Integer> distance = new HashMap<>();
        Queue<Integer> queue = new LinkedList<>();

        distance.put(i * m + j, 0);
        queue.add(i * m + j);

        while (!queue.isEmpty()) {
            int current = queue.poll();
            int x = current / m, y = current % m;
            for (int direction = 0; direction < 4; direction++) {
                int newX = x + deltaX[direction];
                int newY = y + deltaY[direction];
                int next = newX * m + newY;

                if (!isValid(newX, newY, grid)) {
                    continue;
                }

                if (distance.containsKey(next)) {
                    continue;
                }

                distance.put(next, distance.get(current) + 1);

                if (grid[newX][newY] != GridType.HOUSE) {
                    queue.add(next);
                }
            }
        }
        return distance;
    }

    private int getDistanceSum(Map<Integer, Integer> distance, int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int distanceSum = 0;

        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m; y++) {
                int val = grid[x][y];
                if (val == GridType.HOUSE) {
                    if (!distance.containsKey(x * m + y)) {
                        return Integer.MAX_VALUE;
                    }
                    distanceSum += distance.get(x * m + y);
                }
            }
        }

        return distanceSum;
    }

    private boolean isValid(int x, int y, int[][] grid) {
        int n = grid.length, m = grid[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m) {
            return false;
        }

        return grid[x][y] == GridType.EMPTY || grid[x][y] == GridType.HOUSE;
    }

    public String alienOrder(String[] words) {

        // /*
        //indegree of all 26 characters as -1
        Map<Character, Set<Character>> graph = new HashMap<>();
        int[] inDegree = new int[26];
        Arrays.fill(inDegree, -1);

        // build graph and update indegree
        String result = buildGraph(words, graph, inDegree);
        if (result == null) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        //Queue<Character> queue = new LinkedList<>();
        Queue<Character> queue = new PriorityQueue<>();

        //if indegree is 0 put into queue, if is not -1, numOfCharacters++
        int numOfCharacters = 0;
        for (int i = 0; i < 26; i++) {
            if (inDegree[i] == 0) {
                System.out.println((char) (i + 'a'));
                queue.offer((char) (i + 'a'));
            }
            if(inDegree[i]!=-1) {
                numOfCharacters++;
            }
        }

        // iterate queue
        // pop and add to sb
        // if popped is a key in graph, for each of its child, subtract indegree by 1
        // if indegree becomes 0, add to queue
        while (!queue.isEmpty()) {
            char curr = queue.poll();
            System.out.println("queue");
            System.out.println(curr);
            sb.append(curr);
            if (graph.containsKey(curr)) {
                for (char next : graph.get(curr)) {
                    inDegree[next - 'a'] = inDegree[next - 'a']-1;
                    if (inDegree[next - 'a'] == 0) {
                        System.out.println("next");
                        System.out.println(next);
                        queue.offer(next);
                    }
                }
            }
        }

        // return sb, if it's length = numOfCharacters
        return sb.length() == numOfCharacters ? sb.toString() : "";
    }

    // comparing 2 words at a time, and stopping on 1st common letter of those 2
    // and making graph (a Set - kind of adjacency), and indegree
    // put character in first word as KEY, and character of second word for indegree
    // OUT vs IN
    // Set - no particular order but duplicacy control
    private String buildGraph(String[] words, Map<Character, Set<Character>> graph, int[] inDegree) {
        // indegree of all present characters as 0 - on all words
        for (String word : words) {
            for (char c : word.toCharArray()) {
                inDegree[c - 'a'] = 0;
            }
        }
        // 2 pointers - on all words
        for (int i = 1; i < words.length; i++) {
            String first = words[i - 1];
            String second = words[i];

            // min length of 2, to iterate it
            int len = Math.min(first.length(), second.length());
            // iterate it
            for (int j = 0; j < len; j++) {
                char out = first.charAt(j);
                char in = second.charAt(j);
                // stopping on 1st common letter
                if (out != in) {
                    Set<Character> set = graph.getOrDefault(out, new HashSet<>());
                    if (!set.contains(in)) {
                        set.add(in);
                        graph.put(out, set);    // this doesnot put bak, we need to put
                        inDegree[in - 'a']++;
                    }
                    break;      // stopping on first different letter; // Later characters' order are meaningless
                }
                if (j==len-1) {
                    // First = "ab", second = "a" -> invalid
                    if (first.length()> second.length()) {
                        return null;
                    }
                }
            }
        }
        return "";
    }


    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) {
            return null;
        }
        for (RandomListNode node = head; node != null; node = node.next.next) {
            RandomListNode nodeNew = new RandomListNode(node.label);
            nodeNew.next = node.next;
            node.next = nodeNew;
        }
        for (RandomListNode node = head; node != null; node = node.next.next) {
            RandomListNode nodeNew = node.next;
            nodeNew.random = (node.random != null) ? node.random.next : null;
        }
        RandomListNode headNew = head.next;
        for (RandomListNode node = head; node != null; node = node.next) {
            RandomListNode nodeNew = node.next;
            node.next = node.next.next;
            nodeNew.next = (nodeNew.next != null) ? nodeNew.next.next : null;
        }
        return headNew;
    }

    public TreeNode removeNode(TreeNode root, int value) {
        // write your code here
        if (root == null) {
            return null;
        }

        if (value > root.val) {
            root.right = removeNode(root.right, value);
        } else if (value < root.val) {
            root.left = removeNode(root.left, value);
        } else {
            if (root.left == null && root.right == null) {
                root = null;
            } else if (root.right != null) {
                root.val = successor(root);
                root.right = removeNode(root.right, root.val);
            } else {
                root.val = predecessor(root);
                root.left = removeNode(root.left, root.val);
            }
        }

        return root;
    }

    private int successor(TreeNode node) {
        node = node.right;
        while (node.left != null) {
            node = node.left;
        }
        return node.val;
    }

    private int predecessor(TreeNode node) {
        node = node.left;
        while (node.right != null) {
            node = node.right;
        }
        return node.val;
    }


    Stack<TreeNode> upperStack = new Stack<>();
    Stack<TreeNode> lowerStack = new Stack<>();
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        // write your code here
        List<Integer> ans = new ArrayList<>();

        TreeNode current = root;

        while (current != null) {
            upperStack.push(current);
            current = current.left;
        }

        current = root;
        while (current != null) {
            lowerStack.push(current);
            current = current.right;
        }

        while (!upperStack.isEmpty() && upperPeek() < target) {
            moveUpper();
        }

        while (!lowerStack.isEmpty() && lowerPeek() >= target) {
            moveLower();
        }

        for (int i = 0; i < k; i++) {
            if (!upperStack.isEmpty() && !lowerStack.isEmpty()) {
                if (upperPeek() - target < target - lowerPeek()) {
                    ans.add(upperPeek());
                    moveUpper();
                } else {
                    ans.add(lowerPeek());
                    moveLower();
                }
            } else if (!upperStack.isEmpty()) {
                ans.add(upperPeek());
                moveUpper();
            } else {
                ans.add(lowerPeek());
                moveLower();
            }
        }

        return ans;
    }

    private int upperPeek() {
        return upperStack.peek().val;
    }

    private int lowerPeek() {
        return lowerStack.peek().val;
    }

    private void moveUpper() {
        TreeNode current = upperStack.pop();
        current = current.right;
        while (current != null) {
            upperStack.push(current);
            current = current.left;
        }
    }

    private void moveLower() {
        TreeNode current = lowerStack.pop();
        current = current.left;
        while (current != null) {
            lowerStack.push(current);
            current = current.right;
        }
    }

    public static List<List<String>> findLadders(String start, String end, Set<String> dict) {
        // write your code here
        List<List<String>> result = new ArrayList<>();
        Map<String, Integer> distance = new HashMap<>();
        Map<String, List<String>> map = new HashMap<>();
        dict.add(start);
        dict.add(end);

        for (String str : dict) {
            map.put(str, new ArrayList<>());
        }

        findLaddersBFS(end, dict, distance, map);
        List<String> currentList = new ArrayList<>();
        currentList.add(start);
        findLaddersDFS(start, end, distance, map, result, currentList);

        return result;
    }

    private static void findLaddersBFS(String end,
                                Set<String> dict,
                                Map<String, Integer> distance,
                                Map<String, List<String>> map) {

        Queue<String> queue = new LinkedList<>();
        queue.offer(end);
        distance.put(end, 0);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            int currentDistance = distance.get(current);
            List<String> nextList = findNextList(current, dict);
            for (String str : nextList) {
                if (!distance.containsKey(str)) {
                    distance.put(str, currentDistance + 1);
                    queue.offer(str);
                }
                map.get(str).add(current);
            }
        }
    }

    private static List<String> findNextList(String current, Set<String> dict) {
        List<String> result = new ArrayList<>();
        for (int i = 0; i < current.length(); i++) {
            for (char c = 'a'; c <= 'z'; c++) {
                if (current.charAt(i) == c) {
                    continue;
                }

                String next = current.substring(0, i) + c + current.substring(i + 1);
                if (dict.contains(next)) {
                    result.add(next);
                }
            }
        }

        return result;
    }

    private static void findLaddersDFS(String start,
                                String end,
                                Map<String, Integer> distance,
                                Map<String, List<String>> map,
                                List<List<String>> result,
                                List<String> currentList) {

        if (start.equals(end)) {
            result.add(new ArrayList<>(currentList));
            return;
        }

        for (String n : map.get(start)) {
            if (distance.containsKey(n) && distance.get(n) + 1 == distance.get(start)) {
                currentList.add(n);
                findLaddersDFS(n, end, distance, map, result, currentList);
                currentList.remove(currentList.size() - 1);
            }
        }
    }

    public List<String> anagrams(String[] strs) {
        // write your code here
        final int ADDED = -1;
        List<String> ans = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();

        if (strs.length == 0) {
            return ans;
        }

        for (int i = 0; i < strs.length; i++) {
            String str = sortString(strs[i]);

            if (!map.containsKey(str)) {
                map.put(str, i);
            } else {
                int lastWord = map.get(str);
                if (lastWord != ADDED) {
                    ans.add(strs[lastWord]);
                    map.put(str, ADDED);
                }
                ans.add(strs[i]);
            }
        }
        return ans;
    }

    private String sortString(String str) {
        char[] charArr = str.toCharArray();
        Arrays.sort(charArr);
        return Arrays.toString(charArr);
    }


    class cNode {

        public int value;
        public int arrNum;
        public int index;

        public cNode(int value, int arrNum, int index) {
            this.value = value;
            this.arrNum = arrNum;
            this.index = index;
        }

    }

    static Comparator<cNode> compareNode = new Comparator<cNode>() {
        public int compare(cNode o1, cNode o2) {
            return o1.value - o2.value;
        }

    };

    public int[] mergekSortedArrays(int[][] arrays) {
        // write your code here
        PriorityQueue<cNode> queue = new PriorityQueue<>(compareNode);
        List<Integer> ans = new ArrayList<>();

        for (int i = 0; i < arrays.length; i++) {
            if (arrays[i].length == 0) {
                continue;
            }
            queue.add(new cNode(arrays[i][0], i, 0));
        }

        while (!queue.isEmpty()) {
            cNode node = queue.poll();

            int value = node.value;
            int arrIndex = node.arrNum;
            int index = node.index;

            ans.add(value);

            if (index == arrays[arrIndex].length - 1) {
                continue;
            } else {
                queue.add(new cNode(arrays[arrIndex][index + 1], arrIndex, index + 1));
            }
        }

        return ans.stream().mapToInt(Integer::valueOf).toArray();
    }

    public int[] topk(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0) {
            return new int[0];
        }

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int num : nums) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }

        int[] topk = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            topk[i] = minHeap.poll();
        }

        return topk;
    }

    public Point[] kClosest(Point[] points, Point origin, int k) {
        // write your code here
        Arrays.sort(points, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                if ((o1.x - origin.x) * (o1.x - origin.x) + (o1.y - origin.y) * (o1.y - origin.y) == (o2.x - origin.x) * (o2.x - origin.x) + (o2.y - origin.y) * (o2.y - origin.y)) {
                    if (o1.x == o2.x) {
                        return o1.y - o2.y;
                    }
                    return o1.x - o2.x;
                }
                return ((o1.x - origin.x) * (o1.x - origin.x) + (o1.y - origin.y) * (o1.y - origin.y)) - ((o2.x - origin.x) * (o2.x - origin.x) + (o2.y - origin.y) * (o2.y - origin.y));
            }
        });

        return Arrays.copyOfRange(points, 0, k);
    }

}

class RandomListNode {
    int label;
    RandomListNode next, random;
    RandomListNode(int x) { this.label = x; }
};

class GridType {
    static int EMPTY = 0;
    static int HOUSE = 1;
    static int WALL = 2;
}

class AnswerSub {
    int sum;
    int count;

    public AnswerSub(int sum, int count) {
        this.sum = sum;
        this.count = count;
    }
}

class Worker implements Comparable<Worker> {
    public int quality, wage;
    public Worker(int q, int w) {
        quality = q;
        wage = w;
    }

    public double ratio() {
        return (double) wage / quality;
    }

    public int compareTo(Worker other) {
        return Double.compare(ratio(), other.ratio());
    }
}

class ParentTreeNode {
    public ParentTreeNode parent, left, right;

}

class Interval {
    int start, end;
    Interval(int start, int end) {
        this.start = start;
        this.end = end;
    }
}

class LRUCache {

    class DLinkedNode {
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;
        public DLinkedNode() {}
        public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
    }

    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        moveToHead(node);
        return node.value;
    }

    public void set(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode newNode = new DLinkedNode(key, value);
            // 添加进哈希表
            cache.put(key, newNode);
            // 添加至双向链表的头部
            addToHead(newNode);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode tail = removeTail();
                // 删除哈希表中对应的项
                cache.remove(tail.key);
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node.value = value;
            moveToHead(node);
        }
    }

    private void addToHead(DLinkedNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    private DLinkedNode removeTail() {
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
}


class RandomizedSet {
    private Map<Integer, Integer> indexToVal;
    private Map<Integer, Integer> valToIndex;
    public RandomizedSet() {
        // do initialization if necessary
        indexToVal = new HashMap<>();
        valToIndex = new HashMap<>();
    }

    /*
     * @param val: a value to the set
     * @return: true if the set did not already contain the specified element or false
     */
    public boolean insert(int val) {
        // write your code here
        if (valToIndex.containsKey(val)) {
            return false;
        }
        int index = valToIndex.size();
        valToIndex.put(val, index);
        indexToVal.put(index,val);
        return true;
    }


    /*
     * @param val: a value from the set
     * @return: true if the set contained the specified element or false
     */
    public boolean remove(int val) {
        // write your code here
        if (!valToIndex.containsKey(val)) {
            return false;
        }

        int lastVal = indexToVal.get(indexToVal.size() - 1);
        int delIndex = valToIndex.get(val);
        indexToVal.put(delIndex, lastVal);
        valToIndex.put(lastVal, delIndex);
        indexToVal.remove(indexToVal.size() - 1);
        valToIndex.remove(val);
        return true;
    }

    /*
     * @return: Get a random element from the set
     */
    public int getRandom() {
        // write your code here
        return indexToVal.get(new Random().nextInt(indexToVal.size()));
    }
}


class ListNode {
    int val;
    ListNode next;
    ListNode(int x) {
        val = x;
        next = null;
    }
}


class dfsResult {
    int minCost;
    public dfsResult() {
        minCost = Integer.MAX_VALUE;
    }
}

class UndirectedGraphNode {
    int label;
    List<UndirectedGraphNode> neighbors;
    UndirectedGraphNode(int x) {
        label = x;
        neighbors = new ArrayList<UndirectedGraphNode>();
    }
}


class DirectedGraphNode {
    int label;
    List<DirectedGraphNode> neighbors;
    DirectedGraphNode(int x) {
        label = x;
        neighbors = new ArrayList<DirectedGraphNode>();
    }
}


class Point {
    int x;
    int y;

    Point() {
        x = 0;
        y = 0;
    }

    Point(int x, int y) {
        this.x = x;
        this.y = y;
    }
}

class Result {
    int sum;
    int maxSum;
    TreeNode node;

    Result(int sum, int maxSum, TreeNode node) {
        this.sum = sum;
        this.maxSum = maxSum;
        this.node = node;
    }
}




class TreeNode {
    public int val;
    public TreeNode left, right;
    public TreeNode(int val) {
        this.val = val;
        this.left = this.right = null;
    }
 }


class Node {
    Node next;
    int val;

    public Node(int val) {
        this.val = val;
        next = null;
    }
}

class myQueue {
    public Node first = null;
    public Node last = null;
    public void enqueue(int item) {
        // write your code here
        if (first == null) {
            first = new Node(item);
            last = first;
        } else {
            last.next = new Node(item);
            last = last.next;
        }
    }

    /*
     * @return: An integer
     */
    public int dequeue() {
        // write your code here
        if (first == null) {
            return -1;
        }
        int val = first.val;
        first = first.next;
        return val;
    }
}

class Pair{
    int right;
    int left;

    public Pair(int left, int right) {
        this.left = left;
        this.right = right;
    }
}

class TwoSum {

    private List<Integer> nums = new ArrayList<>();
    private Map<Integer, Integer> map = new HashMap<>();
    /**
     * @param number: An integer
     * @return: nothing
     */
    public void add(int number) {
        // write your code here
        if (map.containsKey(number)) {
            map.put(number, map.get(number) + 1);
        } else {
            map.put(number, 1);
            nums.add(number);
        }
    }

    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     */
    public boolean find(int value) {
        // write your code here
        for (int i = 0; i < nums.size(); i++) {
            int num1 = nums.get(i);
            int num2 = value - num1;

            if ((num1 == num2 && map.get(num1) > 1) || (num1 != num2 && map.containsKey(num2))) {
                return true;
            }
        }

        return false;
    }
}

class BSTIterator {
    /**
     * @param root: The root of binary tree.
     */
    private Stack<TreeNode> stack = new Stack<>();
    public BSTIterator(TreeNode root) {
        // do initialization if necessary
        while (root != null) {
            stack.push(root);
            root = root.left;
        }
    }

    /**
     * @return: True if there has next node, or false
     */
    public boolean hasNext() {
        // write your code here
        return !stack.isEmpty();
    }

    /**
     * @return: return next node
     */
    public TreeNode next() {
        // write your code here
        TreeNode curt = stack.peek();
        TreeNode node = curt;

        if (node.right == null) {
            node = stack.pop();
            while (!stack.isEmpty() && stack.peek().right == node) {
                node = stack.pop();
            }
        } else {
            node = node.right;
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
        }
        return curt;
    }

    public int closestValue(TreeNode root, double target) {
        // write your code here
        int closest = root.val;
        while (root != null) {
            if (Math.abs(target - closest) >= Math.abs(root.val - target)) {
                closest = root.val;
            }

            root = root.val < target ? root.right : root.left;
        }

        return closest;
    }

    public int numIslands(boolean[][] grid) {
        // write your code here
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int R = grid.length;
        int C = grid[0].length;
        int count = 0;

        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (grid[i][j]) {
                    count++;
                    grid[i][j] = false;
                    Queue<Integer> queue = new LinkedList<>();
                    queue.add(i * C + j);
                    while (!queue.isEmpty()) {
                        int id = queue.poll();
                        int r = id / C;
                        int c = id % C;
                        if (r - 1 >= 0 && grid[r - 1][c]) {
                            queue.offer((r - 1) * C + c);
                            grid[r - 1][c] = false;
                        }
                        if (r + 1 < R && grid[r + 1][c]) {
                            queue.offer((r + 1) * C + c);
                            grid[r + 1][c] = false;
                        }
                        if (c - 1 >= 0 && grid[r][c - 1]) {
                            queue.offer(r * C + c - 1 );
                            grid[r][c - 1] = false;
                        }
                        if (c + 1 < C && grid[r][c + 1]) {
                            queue.offer(r * C + c + 1);
                            grid[r][c + 1] = false;
                        }
                    }
                }
            }
        }
        return count;
    }

    public List<List<Integer>> subsets(int[] nums) {
        // write your code here
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null) {
            return results;
        }

        if (nums.length == 0) {
            results.add(new ArrayList<>());
            return results;
        }

        Arrays.sort(nums);

        subsetDFS(nums, 0, new ArrayList<>(), results);
        return results;

    }

    private void subsetDFS(int[] nums, int index, List<Integer> subset, List<List<Integer>> results) {
        results.add(new ArrayList<>(subset));

        for (int i = index; i < nums.length; i++) {
            subset.add(nums[i]);
            subsetDFS(nums, i + 1, subset, results);
            subset.remove(subset.size() - 1);
        }
    }
}


class MyQueue {
    Stack<Integer> stack1;
    Stack<Integer> stack2;
    public MyQueue() {
        // do initialization if necessary
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }

    /*
     * @param element: An integer
     * @return: nothing
     */
    public void push(int element) {
        // write your code here
        stack1.push(element);
    }

    /*
     * @return: An integer
     */
    public int pop() {
        // write your code here
        if (!stack2.isEmpty()) {
            return stack2.pop();
        }

        while (!stack1.isEmpty()) {
            stack2.push(stack1.pop());
        }

        return stack2.pop();
    }

    /*
     * @return: An integer
     */
    public int top() {
        // write your code here
        if (!stack2.isEmpty()) {
            return stack2.peek();
        }

        while (!stack1.isEmpty()) {
            stack2.push(stack1.pop());
        }

        return stack2.peek();
    }
}
