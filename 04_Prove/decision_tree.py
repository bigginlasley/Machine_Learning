import math, json

target = "loan"

data = {
    "one": {
        "credit_score": "good",
        "income": "high",
        "collateral": "good",
        "loan": "yes"
    },
    "two": {
        "credit_score": "good",
        "income": "high",
        "collateral": "poor",
        "loan": "yes"
    },
    "three": {
        "credit_score": "good",
        "income": "low",
        "collateral": "good",
        "loan": "yes"
    },
    "four": {
        "credit_score": "good",
        "income": "low",
        "collateral": "poor",
        "loan": "no"
    },
    "five": {
        "credit_score": "average",
        "income": "high",
        "collateral": "good",
        "loan": "yes"
    },
    "six": {
        "credit_score": "average",
        "income": "low",
        "collateral": "poor",
        "loan": "no"
    },
    "seven": {
        "credit_score": "average",
        "income": "high",
        "collateral": "poor",
        "loan": "yes"
    },
    "eight": {
        "credit_score": "average",
        "income": "low",
        "collateral": "good",
        "loan": "no"
    },
    "nine": {
        "credit_score": "low",
        "income": "high",
        "collateral": "good",
        "loan": "yes"
    },
    "ten": {
        "credit_score": "low",
        "income": "high",
        "collateral": "poor",
        "loan": "no"
    },
    "eleven": {
        "credit_score": "low",
        "income": "low",
        "collateral": "good",
        "loan": "no"
    },
    "twleve": {
        "credit_score": "low",
        "income": "low",
        "collateral": "poor",
        "loan": "no"
    }
}

class styles:
    #text colors
    black = '\u001b[30m'
    red = '\u001b[31m'
    green = '\u001b[32m'
    yellow = '\u001b[33m'
    blue = '\u001b[34m'
    magenta = '\u001b[35m'
    cyan = '\u001b[36m'
    white = '\u001b[37m'

    #background colors
    blackBackground = '\u001b[40m'
    redBackground = '\u001b[41m'
    greenBackground = '\u001b[42m'
    yellowBackground = '\u001b[43m'
    blueBackground = '\u001b[44m'
    magentaBackground = '\u001b[45m'
    cyanBackground = '\u001b[46m'
    whiteBackground = '\u001b[47m'

    #extra
    bright = ';1m'
    clear = '\u001b[0m'
    bold = "\u001b[1m"
    underline = "\u001b[4m"
    revers = "\u001b[7m"
style = styles


def calc_data_entropy(data):
    unique_counts = [] 
    entropy_counts = []
    
    # puts the list into a set to eliminate duplicates then grabs each
    for ele in set(data): 
        unique_counts.append(data.count(ele) / len(data))
    
    # calculates entropy
    for i in unique_counts:
        entropy_counts.append(-(i * math.log2(i)))
    return sum(entropy_counts)

# Does the entropy calculation for the whole node (each children's calculations into the overall)
def calc_node_entropy(nodes):
    # calculate entropy for each of the nodes in list and sums them
    entropy = [calc_data_entropy(nodes[i]) for i in range(len(nodes))]
    total = sum([len(nodes[i]) for i in range(len(nodes))])

    # Counts the number of elements in the set from the original data list
    counts = [sum([nodes[i].count(s) for s in set(nodes[i])]) for i in range(len(nodes))]

    # return the total sum of entropy for nodes     
    return sum([entropy[i] * (counts[i]/total) for i in range(len(nodes))])



class Node():
    # constructor, takes in a tag and data, otherwise defaults to none for those variables.
    def __init__(self, tag=None, data=None):
        self.nodes = []
        self.data = data
        self.tag = tag
        self.val = "yes"

    # Populates the nodes 
    def populate_nodes(self, new_questions):
        answers = self.data
        nodelist = []

        # For loop to get the entropy on the given dataset
        for k, v in answers.items():
            # Calls the entropy calculation
            entropy = calc_data_entropy(get_class_from_keys(v, target))

            # if 0.0 then the leaves of that section are all the same, aka done
            if entropy == 0.0:
                nodelist.append(Node(tag = k, data = v))
            # get next node
            else:
                nodelist.append(get_next_node(v, new_questions))
        self.nodes = nodelist

    # Prints out the node and the information based off it's depth
    def format_node(self, depth):
        string = "{}{}{}: {}data{}->{}{}{}\n".format(style.cyan, self.tag, styles.clear, 
                                                        style.red, style.clear, 
                                                        style.green, self.data, style.clear)

        for i in self.nodes:
            string += "   " * depth
            string += i.format_node(depth + 1)

        return string

    # how to represent the nodes, display
    def __repr__(self):
        return self.format_node(0)

# returns the classificaitons from the keys
def get_class_from_keys(keys, target):
    return [data[k][target] for k in keys]

# returns the questions and removes what has been asked
def get_questions_from_test_data(data, target):
    question = list(data[list(data.keys())[0]].keys())
    question.remove(target)
    return question

# Goes through the questions and partions up the data base off the data's classification
def get_same_answers_from_question(question, sub_data):
    i = question
    answers = list(set([val for k, v in sub_data.items() for key, val in v.items() if key == i]))
    answers_dict = {ans: [] for ans in answers}
    [answers_dict[ans].append(k) for ans in answers for k, v in sub_data.items() if ans == v[i]]
    classifiers = [[sub_data[i][target] for i in v] for k, v in answers_dict.items()]
    return classifiers, answers_dict

# Gets all the entropy values
def get_all_entropy_vals(keys, questions, sub_data):
    total = []
    # appends the question and their entropy to a list
    for i in questions:
        classifiers, answers = get_same_answers_from_question(i, sub_data)
        entropy = calc_node_entropy(classifiers)
        total.append([i, entropy])
    # sorts the list of questions by their entropy 
    total.sort(key=lambda x: x[1])
    return total

# The main function that is called to build the node for the tree
def get_next_node(keys, questions):
    data_entropy = calc_data_entropy(get_class_from_keys(keys, target))

    # If all leavs are the same end
    if data_entropy == 0:
        return Node(tag="ent 0", data=keys)
    # Otherwise continue
    else:
        # If we are out of questions we hit break case
        if not questions:
            return Node(tag="no questions", data=keys)
        # Otherwise continue 
        else:
            # Find which questiton gives most information gain
            sub_data = {k:data[k] for k in keys}
            sub_ents = get_all_entropy_vals(keys, questions, sub_data)

            lowest, lowest_ent = sub_ents[0]

            # Gets the questions that need to be removed
            new_questions = [i for i in questions if not i == lowest]

            answers = get_same_answers_from_question(lowest, sub_data)[1]

            # Returns the node that we have buit from the algorithm
            n = Node(tag=lowest, data=answers)
            n.populate_nodes(new_questions)
            return n
    # build tree node by node
def buildTree():
    all_keys = list(data.keys())
    all_questions = get_questions_from_test_data(data, target)
    print(get_next_node(all_keys, all_questions))

# main function
def main():
    buildTree()


# runs main
if __name__ =="__main__":
    main()

 