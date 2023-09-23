import java.util.List;

public class MainMac {
    public static void main (String[] args) {
    }

    public void heapify(int[] a) {
        // write your code here
        for (int i = 0; i < a.length; i++) {
            heapifyHelper(a, i);
        }
    }

    private void heapifyHelper(int[] a, int index) {
        while (index != 0) {
            int father = (index - 1) / 2;
            if (a[index] > a[father]) {
                break;
            }

            int temp = a[father];
            a[father] = a[index];
            a[index] = temp;

            index = father;
        }
    }

    public void sortIntegers(int[] a) {
        // write your code here
        int length = a.length;
        for (int i = length - 1; i >= 0 ; i--) {
            boolean isSorted = true;
            for (int j = 0; j < i; j++) {
                if (a[j] > a[j + 1]) {
                    bubbleSwap(a, j, j + 1);
                    isSorted = false;
                }
            }
            if (isSorted) {
                break;
            }
        }
    }

    private void bubbleSwap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }

    public TreeNode findSubtree(TreeNode root) {
        // write your code here
        ResultMac result = subtreeHelper(root);
        return result.minNode;
    }

    private ResultMac subtreeHelper(TreeNode node) {
        if (node == null) {
            return new ResultMac(0, Integer.MAX_VALUE, null);
        }

        ResultMac leftResult = subtreeHelper(node.left);
        ResultMac rightResult = subtreeHelper(node.right);

        int sum = leftResult.sum + rightResult.sum + node.val;

        ResultMac result = new ResultMac(sum, sum, node);

        if (result.minSum > leftResult.minSum) {
            result.minNode = leftResult.minNode;
            result.minSum = leftResult.minSum;
        }

        if (result.minSum > rightResult.minSum) {
            result.minNode = rightResult.minNode;
            result.minSum = rightResult.minSum;
        }

        return result;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode A, TreeNode B) {
        // write your code here
        if (root == null) {
            return null;
        }

        if (root.val == A.val || root.val == B.val) {
            return root;
        }

        TreeNode left = lowestCommonAncestor(root.left, A, B);
        TreeNode right = lowestCommonAncestor(root.right, A, B);

        if (left != null && right != null) {
            return root;
        }

        if (left!= null) {
            return left;
        }

        if (right != null) {
            return right;
        }

        return null;
    }

    public int longestConsecutive(TreeNode root) {
        // write your code here
        return longestHelper1(root, null, 0);
    }

    private int longestHelper1(TreeNode root, TreeNode parent, int length) {
        if (root == null) {
            return length;
        }

        if (parent != null && parent.val + 1 == root.val) {
            length++;
        } else {
            length = 1;
        }

        return Math.max(length, Math.max(longestHelper1(root.left, root, length),
                                            longestHelper1(root.right, root, length)));
    }

    public int longestConsecutive2(TreeNode root) {
        LongestResult2 result = longestHelper2(root);
        return result.longestConsec;
    }

    private LongestResult2 longestHelper2(TreeNode node) {
        if (node == null) {
            return new LongestResult2(0, 0, 0);
        }


        LongestResult2 leftResult = longestHelper2(node.left);
        LongestResult2 rightResult = longestHelper2(node.right);

        int leftIncrease = leftResult.curIncrease;
        int leftDecrease = leftResult.curDecrease;
        int rightIncrease = rightResult.curIncrease;
        int rightDecrease = rightResult.curDecrease;
        int curIncrease = 0;
        int curDecrease = 0;

        if (node.left != null) {
            if (node.val == node.left.val + 1) {
                curDecrease = Math.max(curDecrease, leftDecrease + 1);
            }
            if (node.val == node.left.val - 1) {
                curIncrease = Math.max(curIncrease, leftIncrease + 1);
            }
        }


        if (node.right != null) {
            if (node.val == node.right.val + 1) {
                curDecrease = Math.max(curDecrease, rightDecrease + 1);
            }
            if (node.val == node.right.val - 1) {
                curIncrease = Math.max(curIncrease, rightIncrease + 1);
            }
        }

        int longest = Math.max(Math.max(leftResult.longestConsec, rightResult.longestConsec),
                                curDecrease + curIncrease + 1);

        return new LongestResult2(curIncrease, curDecrease, longest);
    }





    public int longestConsecutive3(MultiTreeNode root) {
        // Write your code here
        LongestResult2 result = longestHelper3(root);
        return result.longestConsec;
    }

    private LongestResult2 longestHelper3(MultiTreeNode node) {
        if (node == null) {
            return new LongestResult2(0, 0, 0);
        }

        int curIncrease = 0, curDecrease = 0, longest = 0;
        for (MultiTreeNode child : node.children) {
            LongestResult2 result = longestHelper3(child);

            if (node.val == child.val + 1) {
                curDecrease = Math.max(curDecrease, result.curDecrease + 1);
            }
            if (node.val == child.val - 1) {
                curIncrease = Math.max(curIncrease, result.curIncrease + 1);
            }

            longest = Math.max(longest, result.longestConsec);
        }

        longest = Math.max(longest, curIncrease + curDecrease + 1);

        return new LongestResult2(curIncrease, curDecrease, longest);

    }



}

class LongestResult2 {
    int curIncrease, curDecrease, longestConsec;

    public LongestResult2(int curIncrease, int curDecrease, int longestConsec) {
        this.curIncrease = curIncrease;
        this.curDecrease = curDecrease;
        this.longestConsec = longestConsec;
    }
}


class MultiTreeNode {
    int val;
    List<MultiTreeNode> children;
    MultiTreeNode(int x) { val = x; }
}

class ResultMac {
    int sum, minSum;
    TreeNode minNode;

    public ResultMac (int sum, int minSum, TreeNode node) {
        this.minNode = node;
        this.sum = sum;
        this.minSum = minSum;
    }
}


