import java.util.*;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    public static void main(String[] args) {
        kthLargestElement(2, new int[]{1, 2, 5, 3, 4});
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
}


class RandomizedSet {
    private Map<Integer, Integer> indexToVal;
    private Map<Integer, Integer> valToIndex;
    public RandomizedSet() {
        // do intialization if necessary
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
