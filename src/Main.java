import sun.awt.image.ImageWatched;

import java.util.*;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    public static void main(String[] args) {
        int[] A = {1,2,3};
        int[] B = {4, 5};
        mergeSortedArray(A, 3, B, 2);
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
