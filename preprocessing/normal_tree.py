from nltk.tree import Tree
import re


class NormalTree(Tree):
    """
    A normal tree is a tree that:
    1. has no empty label or TOP at the root
    2. has no functional labels
    3. has no '-NONE-' branches
    4. has no X->X productions
    5. is binarized
    6. leaves are all lower-cased
    """
    @staticmethod
    def remove_functional(label):
        """
        Removes functional labels.
        """
        cut_index = label.find('-')
        cut_index2 = label.find('=')
        if cut_index2 > 0 and (cut_index2 < cut_index or cut_index <= 0):
            cut_index = cut_index2
        if cut_index > 0:
            label = label[:cut_index]
        return label
    
    @staticmethod
    def remove_identity(tree):
        """
        This function is a modified version of nltk.treetransforms.collapse_unary.
        It removes X->X productions.
        """
        if not isinstance(tree, Tree):
            raise RuntimeError('Not a valid tree.')
        nodeList = [tree]

        # depth-first traversal of tree
        while nodeList != []:
            node = nodeList.pop()
            if (len(node) == 1
                and isinstance(node[0], Tree)
                and (node.label() == node[0].label())
               ):
                node[0:] = [child for child in node[0]]
                nodeList.append(node)
            else:
                for child in node:
                    if not isinstance(child, Tree):
                        continue
                    nodeList.append(child)
    
    @staticmethod
    def remove_empty(node):
        """
        Removes '-NONE-'s.
        """
        if not isinstance(node, Tree):
            return node
        if node.label() == '-NONE-':
            return None
        new_children = []
        for child in node:
            new_child = NormalTree.remove_empty(child)
            if new_child is not None:
                new_children.append(new_child)
        if len(new_children) == 0:
            return None
        return NormalTree(node.label(), new_children)

    @staticmethod
    def to_lower(node):
        if len(node) == 1 and isinstance(node[0], str):
            node[0] = node[0].lower()
        else:
            for subtree in node:
                NormalTree.to_lower(subtree)
    
    @classmethod
    def normal_fromstring(cls, s):
        """
        This function is a modified version of nltk.tree.Tree.fromstring.
        """
        brackets = "()"
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        node_pattern = "[^\s%s%s]+" % (open_pattern, close_pattern)
        leaf_pattern = "[^\s%s%s]+" % (open_pattern, close_pattern)
        token_re = re.compile(
            "%s\s*(%s)?|%s|(%s)"
            % (open_pattern, node_pattern, close_pattern, leaf_pattern)
        )
        
        stack = [(None, [])]
        for match in token_re.finditer(s):
            token = match.group()
            # Beginning of a tree/subtree
            if token[0] == open_b:
                if len(stack) == 1 and len(stack[0][1]) > 0:
                    cls._parse_error(s, match, "end-of-string")
                label = token[1:].lstrip()
                label = cls.remove_functional(label)
                stack.append((label, []))
            # End of a tree/subtree
            elif token == close_b:
                if len(stack) == 1:
                    if len(stack[0][1]) == 0:
                        cls._parse_error(s, match, open_b)
                    else:
                        cls._parse_error(s, match, "end-of-string")
                label, children = stack.pop()
                stack[-1][1].append(cls(label, children))
            # Leaf node
            else:
                if len(stack) == 1:
                    cls._parse_error(s, match, open_b)
                stack[-1][1].append(token)

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            cls._parse_error(s, "end-of-string", close_b)
        elif len(stack[0][1]) == 0:
            cls._parse_error(s, "end-of-string", open_b)
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1
        tree = stack[0][1][0]

        # Remove top empty brakcet or TOP
        if tree._label == "" or tree._label == 'TOP':
            tree = tree[0]
        tree = cls.remove_empty(tree)
        cls.remove_identity(tree)
        tree.chomsky_normal_form(factor='left', horzMarkov=0, vertMarkov=0)
        tree.collapse_unary(collapseRoot=True, collapsePOS=True)
        cls.to_lower(tree)
        return tree
