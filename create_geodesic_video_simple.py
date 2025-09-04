import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import eigh
import sys
import os

# Add the randomcov package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'randomcov'))

from randomcov.covutil.geodesicinterpolation import geodesic_interpolation_towards_perfect

# Set style for beautiful plots
plt.style.use('dark_background')
sns.set_palette("husl")

def create_covariance_matrix(n=4):
    """Create a random positive definite covariance matrix"""
    # No fixed seed - will generate different matrices each time
    # Generate random matrix
    A = np.random.randn(n, n)
    # Make it symmetric and positive definite
    cov = A @ A.T + n * np.eye(n)
    # Normalize to have reasonable values
    cov = cov / np.max(np.abs(cov)) * 5
    return cov

def plot_3d_matrix(ax, matrix, title, cmap='viridis'):
    """Create a 3D surface plot of a matrix"""
    n = matrix.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    # Create the surface
    surf = ax.plot_surface(X, Y, matrix, cmap=cmap, 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Row Index', color='white')
    ax.set_ylabel('Column Index', color='white')
    ax.set_zlabel('Value', color='white')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    
    # Set view angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    # Style the axes
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    return surf

def plot_2d_covariance_bars(ax, matrix, title):
    """Create a 2D bar plot of covariance matrix elements"""
    n = matrix.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the matrices for plotting
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = matrix.flatten()
    
    # Create bar plot
    bars = ax.bar3d(x_flat, y_flat, np.zeros_like(z_flat), 
                    0.8, 0.8, z_flat, 
                    color=plt.cm.viridis((z_flat - z_flat.min()) / (z_flat.max() - z_flat.min())),
                    alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Row Index', color='white', fontsize=12)
    ax.set_ylabel('Column Index', color='white', fontsize=12)
    ax.set_zlabel('Covariance Value', color='white', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    
    # Set view angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    # Style the axes
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Set limits
    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.5, n-0.5)
    
    return bars

def calculate_minimum_variance_portfolio(cov_matrix):
    """Calculate the minimum variance portfolio weights"""
    n = cov_matrix.shape[0]
    
    # Add constraint: sum of weights = 1
    A = np.vstack([np.ones(n), np.eye(n)])
    b = np.array([1.0] + [0.0] * n)
    
    # Solve the quadratic programming problem
    try:
        # Use pseudo-inverse for numerical stability
        cov_inv = np.linalg.pinv(cov_matrix)
        weights = cov_inv @ np.ones(n)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # Ensure weights are reasonable
        weights = np.clip(weights, -0.5, 1.5)
        weights = weights / np.sum(weights)
        
        return weights
    except:
        # Fallback to equal weights if there's an issue
        return np.ones(n) / n

def plot_portfolio_weights(ax, weights, title):
    """Plot portfolio weights as a bar chart"""
    n = len(weights)
    x = np.arange(n)
    
    # Create bar plot
    colors = plt.cm.plasma((weights - weights.min()) / (weights.max() - weights.min()))
    bars = ax.bar(x, weights, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', 
                color='white', fontweight='bold', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Asset Index', color='white', fontsize=12)
    ax.set_ylabel('Portfolio Weight', color='white', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Asset {i+1}' for i in range(n)])
    
    # Style the axes
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='white', linestyle='-', alpha=0.5)
    
    return bars

def create_sexy_animation():
    """Create a sexy animation with beautiful visualizations"""
    # Create figure with sophisticated layout
    fig = plt.figure(figsize=(24, 12))
    
    # Create grid layout (removed top row, adjusted height ratios)
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.3], width_ratios=[1, 1, 1, 1, 1, 1])
    
    # 2D bar plots for covariance matrices (first row)
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    ax2 = fig.add_subplot(gs[0, 2:4], projection='3d')
    ax3 = fig.add_subplot(gs[0, 4:6], projection='3d')
    
    # Portfolio weights and eigenvalue evolution (second row)
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])
    
    # Progress bar and metrics (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create initial covariance matrix
    original_cov = create_covariance_matrix(n=4)
    
    # Calculate perfect correlation matrix
    std_devs = np.sqrt(np.diag(original_cov))
    perfect_cov = np.outer(std_devs, std_devs) * 0.99
    np.fill_diagonal(perfect_cov, np.diag(original_cov))
    
    # Store eigenvalues for plotting
    eigenvals_original, _ = eigh(original_cov)
    eigenvals_perfect, _ = eigh(perfect_cov)
    
    # Animation function
    def animate(frame):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        
        # Calculate gamma (interpolation parameter)
        gamma = frame / 50.0  # Fewer frames for GIF
        
        # Get interpolated matrix
        interpolated_cov = geodesic_interpolation_towards_perfect(original_cov, gamma)
        eigenvals_interp, _ = eigh(interpolated_cov)
        
        # Plot 2D bar plots for covariance matrices
        plot_2d_covariance_bars(ax1, original_cov, f'Original Covariance\n(γ = 0.00)')
        plot_2d_covariance_bars(ax2, interpolated_cov, f'Interpolated Covariance\n(γ = {gamma:.2f})')
        plot_2d_covariance_bars(ax3, perfect_cov, f'Perfect Correlation\n(γ = 1.00)')
        
        # Plot minimum variance portfolio weights
        weights_original = calculate_minimum_variance_portfolio(original_cov)
        weights_interp = calculate_minimum_variance_portfolio(interpolated_cov)
        weights_perfect = calculate_minimum_variance_portfolio(perfect_cov)
        
        plot_portfolio_weights(ax4, weights_interp, f'Minimum Variance Portfolio Weights\n(γ = {gamma:.2f})')
        
        # Plot eigenvalue evolution
        ax5.plot([0, gamma, 1], [eigenvals_original[0], eigenvals_interp[0], eigenvals_perfect[0]], 
                'o-', linewidth=3, markersize=8, label='λ₁', color='red')
        ax5.plot([0, gamma, 1], [eigenvals_original[1], eigenvals_interp[1], eigenvals_perfect[1]], 
                'o-', linewidth=3, markersize=8, label='λ₂', color='blue')
        ax5.plot([0, gamma, 1], [eigenvals_original[2], eigenvals_interp[2], eigenvals_perfect[2]], 
                'o-', linewidth=3, markersize=8, label='λ₃', color='green')
        ax5.plot([0, gamma, 1], [eigenvals_original[3], eigenvals_interp[3], eigenvals_perfect[3]], 
                'o-', linewidth=3, markersize=8, label='λ₄', color='orange')
        ax5.set_xlabel('Interpolation Parameter γ', color='white', fontsize=12)
        ax5.set_ylabel('Eigenvalues', color='white', fontsize=12)
        ax5.set_title('Eigenvalue Evolution Along Geodesic', color='white', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 1)
        ax5.tick_params(colors='white')
        
        # Progress bar and metrics
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        # Progress bar
        progress_bar = plt.Rectangle((0.1, 0.4), gamma * 0.8, 0.2, facecolor='cyan', alpha=0.8)
        ax6.add_patch(progress_bar)
        ax6.add_patch(plt.Rectangle((0.1, 0.4), 0.8, 0.2, facecolor='none', edgecolor='white', linewidth=2))
        
        # Progress text
        ax6.text(0.5, 0.7, f'TRANSFORMATION PROGRESS: {gamma*100:.1f}%', 
                ha='center', fontsize=16, fontweight='bold', color='cyan')
        ax6.text(0.5, 0.2, f'γ = {gamma:.3f} | Portfolio Variance: {weights_interp @ interpolated_cov @ weights_interp:.3f}', 
                ha='center', fontsize=12, color='white')
        
        # Add main title
        fig.suptitle('🎯 GEODESIC INTERPOLATION TOWARDS PERFECT CORRELATION 🎯\n'
                    'Covariance Evolution & Portfolio Optimization', 
                    fontsize=24, fontweight='bold', color='white', y=0.98)
        
        # Add explanation text
        fig.text(0.5, 0.01, 
                'This visualization shows how covariance matrices transform along geodesic paths using 2D bar plots, '
                'how minimum variance portfolio weights evolve, and eigenvalue changes. The transformation preserves '
                'positive definiteness while smoothly interpolating towards perfect correlation structure.', 
                ha='center', fontsize=12, color='lightgray', style='italic')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=51, interval=200, repeat=True)
    
    return anim

if __name__ == "__main__":
    print("🎬 Creating sexy geodesic interpolation video...")
    print("🚀 Generating advanced animation with 3D visualizations...")
    
    # Create the sexy animation
    anim = create_sexy_animation()
    
    print("💾 Saving animation as 'geodesic_interpolation_sexy.gif'...")
    
    # Save the animation as GIF
    anim.save('geodesic_interpolation_sexy.gif', writer='pillow', fps=5, dpi=100)
    
    print("✅ GIF saved successfully!")
    print("🎉 Your sexy geodesic interpolation animation is ready!")
    print("\n📊 The animation shows:")
    print("   • 2D bar plots of covariance matrices (top row)")
    print("   • Minimum variance portfolio weights (bottom left)")
    print("   • Eigenvalue evolution along the geodesic (bottom right)")
    print("   • Real-time progress tracking")
    print("   • Beautiful mathematical aesthetics")
    
    # Display the animation
    print("🎬 Animation created successfully!")
    print("📱 If the animation doesn't display, check the generated GIF file:")
    print("   - Look for 'geodesic_interpolation_sexy.gif' in the current directory")
    print("   - Open it with any image viewer or web browser")
    
    # Try to display the animation
    try:
        plt.show()
        print("✅ Animation displayed successfully!")
    except Exception as e:
        print(f"⚠️  Could not display animation: {e}")
        print("💡 The GIF file was still created and can be viewed separately")
