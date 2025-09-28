import matplotlib.pyplot as plt

def plot_model_metrics(metrics_dict, title="Métricas de Avaliação", save_path="plot/metrics_plot.png"):
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_dict.keys(), metrics_dict.values(), color='skyblue')
    plt.title(title)
    plt.ylabel('Erro')
    for i, v in enumerate(metrics_dict.values()):
        plt.text(i, v + max(metrics_dict.values())*0.01, f"{v:.2f}", ha='center')
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Gráfico salvo em: {save_path}")
    plt.close()  # fecha a figura para liberar memória
