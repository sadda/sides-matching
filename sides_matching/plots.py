import matplotlib.pyplot as plt

def plot1(analysis, similarity_split, i_diff, dataset_name=None, file_name=None, ylim=None):
    fig = plt.figure()
    for j, name_split in enumerate(['database-database', 'database-query', 'query-query']):
        ax = fig.add_subplot(1, 3, j+1)
        similarity_boxplot = []
        for name in analysis.names_categories:
            similarity_boxplot.append(similarity_split[name_split][i_diff][name])

        plt.boxplot(similarity_boxplot)
        ax.set_xticklabels(analysis.names_categories)
        #ax.set_aspect(8)
        plt.xticks(rotation=25)
        if ylim is not None:
            plt.ylim(ylim)
        plt.axhline(y=0, color='r', linestyle='dotted')                
        if j == 0:
            plt.ylabel('similarity')
            ax.set_title(f' \n {name_split}')
        elif j == 1:
            ax.set_title(f'{dataset_name}: {analysis.diff_to_matches[i_diff]} \n {name_split}')
        elif j == 2:
            ax.set_title(f' \n {name_split}')
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

def plot2(analysis, similarity_split, name_category, dataset_name=None, file_name=None, ylim=None):
    fig = plt.figure()
    for j, name_split in enumerate(['database-database', 'database-query', 'query-query']):    
        ax = fig.add_subplot(1, 3, j+1)
        similarity_boxplot = []
        for i in range(analysis.max_diff+1):
            similarity_boxplot.append(similarity_split[name_split][i][name_category])

        plt.boxplot(similarity_boxplot)
        ax.set_xticklabels(analysis.diff_to_matches.values())
        #ax.set_aspect(8)
        plt.xticks(rotation=25)
        if ylim is not None:
            plt.ylim(ylim)
        plt.axhline(y=0, color='r', linestyle='dotted')
        if j == 0:
            plt.ylabel('similarity')
            ax.set_title(f' \n {name_split}')
        elif j == 1:
            ax.set_title(f'{dataset_name}: {name_category} \n {name_split}')
        elif j == 2:
            ax.set_title(f' \n {name_split}')
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)