import java.util.*;
import java.util.stream.Collectors;

public class topK {
    /*
     * @param k: An integer
     */
    public int k;
    public PriorityQueue<Integer> minHeap;

    public topK(int k) {
        // do initialization if necessary
        this.k = k;
        this.minHeap = new PriorityQueue<>();
    }

    /*
     * @param num: Number to be added
     * @return: nothing
     */
    public void add(int num) {
        // write your code here
        minHeap.offer(num);
        if (minHeap.size() > k) {
            minHeap.poll();
        }
    }

    /*
     * @return: Top k element
     */
    public List<Integer> topk() {
        // write your code here

        int size = minHeap.size();

        int[] ans = new int[size];

        for (int i = size - 1; i >= 0; i--) {
            ans[i] = minHeap.poll();
        }

        for (int i = 0; i < size; i++) {
            minHeap.add(ans[i]);
        }

        return Arrays.stream(ans).boxed().collect(Collectors.toList());
    }
}

class Solution1 {
    int[][] dir = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    public int wordSearchIII(char[][] board, List<String> words) {
        // write your code here
        Trie trie = new Trie();for (String word : words) {
            trie.insert(word);
        }

        // 答案放在数组里，作为reference往下传
        int[] ans = new int[1];
        ans[0] = 0;
        int n = board.length, m = board[0].length;
        // visited记录访问过的位置
        boolean[][] visited = new boolean[n][m];

        // 从board上的每一个点出发，看能否找到最多的单词
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                // i，j是坐标，需要把trie也传进去，
                // 因为当上面情况1的时候（找到单词），需要从root开始搜索
                dfs(i, j, board, visited, trie.root, trie, 0, ans);
            }
        }
        return ans[0];
    }

    private void dfs(int x, int y, char[][] board, boolean[][] visited, TrieNode node, Trie trie, int wordCount, int[] ans) {
        // 如果字典树不包含board上扩展的字母，直接返回
        if (!node.children.containsKey(board[x][y])) return;
        // 若字典树包含board扩展的字母，记录当前位置被访问过
        visited[x][y] = true;
        // 字典树也需要走到board的字母位置，找到当前node
        TrieNode child = node.children.get(board[x][y]);
        // 如果当前是一个单词
        if (child.isWord) {
            wordCount++;
            // 把isword标记false，以防止将来重复搜索为单词
            child.isWord = false;
            // 打擂台找最大单词数
            ans[0] = Math.max(ans[0], wordCount);

            // 若当前是单词停止遍历，返回0，0位置，继续找下一个单词
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[0].length; j++) {
                    if (visited[i][j]) continue;
                    dfs(i, j, board, visited, trie.root, trie, wordCount, ans);
                }
            }
            // 遍历完了，再回溯标记为true，以供下次遍历别的路径使用
            child.isWord = true;
            // 走到这里，当前单词就是以前缀方式出现了，所以wordcount-1
            wordCount--;
        }

        // 以当前字母为前缀append，继续查找四个方向
        for (int i = 0; i < 4; i++) {
            int nx = x + dir[i][0], ny = y + dir[i][1];
            if (!isValid(nx, ny, visited)) continue;
            // 四个方向递归看一下
            dfs(nx, ny, board, visited, child, trie, wordCount, ans);
        }

        // 找完记得回溯，标记为没访问过，
        // 因为从其他地方出发的点，可能用上当前的字母
        visited[x][y] = false;
    }

    // 看是否越界，和被访问过
    private boolean isValid(int x, int y, boolean[][] visited) {
        int n = visited.length, m = visited[0].length;
        if (x < 0 || x >= n || y < 0 || y >= m) return false;
        return visited[x][y] == false;
    }
}

class TrieNode {
    Map<Character, TrieNode> children;
    String word;
    boolean isWord;
    public TrieNode() {
        children = new HashMap<>();
        word = null;
        isWord = false;
    }
}

class Trie {
    TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (!node.children.containsKey(c)) {
                node.children.put(c, new TrieNode());
            }
            node = node.children.get(c);
        }

        node.word = word;
        node.isWord = true;
    }
}