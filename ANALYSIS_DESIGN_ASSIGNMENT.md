# Object Detection Case Study Assignment

## Objectives

This assignment will guide you in evaluating the **TechTrack System** by:

* Comparing object-detection models
* Analyzing the effects of specific augmentations on model performance
* Exploring how Hard Negative Mining (HNM) parameters influence sample selection
* Submit a file (`<BASE_DIRECTORY>/CASE_ANALYSIS.md`) to summarize your findings.
* Submit supporting Jupyter notebooks (`<BASE_DIRECTORY>/analysis/task*.ipynb`) to show your work.

## Tasks

Please compile all your findings—including every table, figure, and succinct commentary—into a single, cohesive document (`CASE_ANALYSIS.md`).

* Each **table** and **figure** must be clearly labeled (e.g., *Table 1: Model Comparison Metrics*, *Figure 2: PR Curves*).
* Captions should concisely describe what the reader should observe.
* After each visualization or table, provide a **brief interpretation** explaining the key takeaway (e.g., *Table 3 shows a 7‑point drop in mAP when brightness is reduced by 50%*), and explain how this finding/insight would impact your design choices of your system.
* Do **NOT** include step-by-step code. 

Maintain consistent formatting throughout: uniform fonts, spacing, numbering, and captions. Ensure the document reads smoothly with explanations that remind the reader **how  and why each analysis was performed (methodology)** and **how your results supports your overall conclusion**.

You will be evaluated on:

* Robustness of your methodology and reasoning (are your conclusions well-supported by evidence? and is your reasoning correct?)
* Depth and completeness of your analysis (did you explore trade-offs and alternative explanations?)
* Clarity and cohesion of your presentation (does your report read smoothly as a logical narrative?)

For each of the five tasks, include a dedicated Jupyter Notebook (`analysis/task1.ipynb`, `analysis/task2.ipynb`, …, and `analysis/task5.ipynb`) showing your analysis, intermediate exploration, and any code used. Each notebook should clearly demonstrate how you arrived at your reported results, even if the final report only summarizes them. The evaluation will not consider the contents contained in these notebooks. These will only be used to clarify details missing in your markdown file, if need be.

### Task Breakdown

1. **Model Performance Comparison (Model Selection)**: Compare the overall performance of Model 1 and Model 2 on the full TechTrack dataset. Identify which classes one model handles better than the other.

💡 **Hint:** The complete dataset is large (though small compared to real-world datasets). To improve efficiency, precompute and save model outputs to disk. Also, you are encouraged to use Jypyter Notebooks.

2. **Dataset Sampling Strategy (Dataset Design)**: Propose and describe a clear sampling strategy for working with the TechTrack dataset. You may use your proposed dataset to complete the remaining tasks. You must use a minimum of five thousand images (or more, depending on your individual compute capabilities). Your proposal should include:

   * Criteria for selecting representative subsets of data
   * Justification for why this sampling strategy is valid

3. **Threshold Design (Parameter Configuration)**: Using your best performing model (Task 1), decide on and argue for the best threshold value of your NMS module. 

4. **Augmentation Impact (Robustness Analysis)**: Measure the effects of Gaussian Blur, Vertical Flips, and Brightness adjustments on your best performing model's performance (Task 1).

5. **HNM Sampling Strategy (Parameter Configuration)**: Analyze how different $\lambda$ values in HNM affect sampling of images. Show and describe how varying these parameters would impact the types of images sampled.


## Submission

* Commit and push your **`CASE_ANALYSIS.md`** file and your supporting notebooks (e.g., `analysis/task*.ipynb`) into your provisioned GitHub repository.
* Provide your **repository URL** when submitting.


