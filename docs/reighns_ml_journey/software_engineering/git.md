<div align="center">
<h1>Git Tutorial (Based on Atlassian)</a></h1>
<br>
</div>

---

The following notes' taken from [Colt Steele](https://www.youtube.com/watch?v=USjZcfj8yxE). However, I have added some steps on remote repositories towards the end.

For ease of use, this tutorial assumes one works on macOS but a 1-1 mapping of commands is available for Windows and Linux.

A typical workflow

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <>
git push -u origin master
```

## Installation and Configuration

### Installing Git

We have to ensure that we have Git installed on our machine. 
Different operating systems have different ways of installing Git.
The following command downloads Git to macOS:

```bash title="install git" linenums="1"
~/gaohn $ brew install git # (1)
```

1. Use Homebrew to install git manager.

Once Git is installed, we can check the version of Git we have installed:

```bash title="check git version" linenums="1"
~/gaohn $ git --version 
```

For more detailed instructions, you can follow the [Atlassian's Install
Git Tutorial](https://www.atlassian.com/git/tutorials/install-git).

---

### Configure Name and Email

In your terminal, configure your **Git username** and **email** using the following commands: 

```bash title="configure git" linenums="1"
git config --global user.name {gaohn}
git config --global user.email {gaohn@gmail.com}
```

replacing `gaohn` with your own name, and set the `user.email` to the email address you used to sign up for GitHub.

These details will be associated with any commits that you create!

!!! note
    This step is particularly important if you are working with multiple git accounts.
    This is because setting the config globally will affect your commits.
    
    For example, if I have two git accounts called `a1` and `a2` with emails
    `a1@gmail.com` and `a2@gmail.com` respectively. If I set my global config
    of the `user.email` to `a1@gmail.com`, then any commits made by `a2` will
    be credited to `a1` as contributor.

---

## Getting Started

### Setting Up Repository

We will closely follow the [Atlassian's Getting Started Tutorial](https://www.atlassian.com/git/tutorials/setting-up-a-repository).

As stated in their tutorial, this section's high level overview:

- Initializing a new Git repo
- Cloning an existing Git repo
- Committing a modified version of a file to the repo
- Configuring a Git repo for remote collaboration
- Common Git version control commands

#### Initialize an Entirely New Repository: `git init`

To create a new repo, you'll use the `git init` command;
this a one-time command you use during the initial setup of a new repo. 

Executing this command will create a new `.git` subdirectory in your current working directory. 

This will also create a new `main/master` branch. 

```bash title="initialize a new repo" linenums="1"
~/gaohn              $ mkdir {git_tutorial}
~/gaohn              $ cd {git_tutorial}
~/gaohn/git_tutorial $ git init

> Initialized empty Git repository in /Users/gaohn/git_tutorial/.git/
```

This command will generate a hidden **.git** directory for your project, where Git stores all internal tracking data for the current repository.

```bash
~/gaohn/git_tutorial $ ls -a
```

Visit the [Atlassian's git init tutorial](https://www.atlassian.com/
git/tutorials/setting-up-a-repository/git-init) for more details.

---

#### Initialize a Repository from an Existing Project: `git clone`

If a project has already been set up in a ***remote repository***, the clone command is the most 
common way for users to obtain a local development clone. 

Like `git init`, cloning is generally a one-time operation. Once a developer has obtained a working 
copy, all version control operations are managed through their local repository.

```bash title="clone a repo" linenums="1"
~/gaohn $ git clone <repo url>
```

For example, I have a repository stored on GitHub called [https://github.com/ghnreigns/
github-test.git](https://github.com/ghnreigns/github-test.git) and I can clone it to my 
local machine by filling `<repo url>` with the url of the repository.

```bash title="clone a repo" linenums="1"
~/gaohn $ git clone https://github.com/ghnreigns/github-test.git

> Cloning into 'github-test'...
> remote: Enumerating objects: 6, done.
> remote: Counting objects: 100% (6/6), done.
> remote: Compressing objects: 100% (2/2), done.
> remote: Total 6 (delta 0), reused 0 (delta 0), pack-reused 0
> Receiving objects: 100% (6/6), done.
```

Then you will see a new directory called `github-test` in your current working directory and 
you do not need to run `git init` on this directory.

These two commands cover the ground of initializing a repository locally.

Visit the [Atlassian's git clone tutorial](https://www.atlassian.com/git/tutorials/
setting-up-a-repository/git-clone) for more details.

#### Initialize Remote Repository

Generally stored outside of your isolated local system, usually on a remote server. It's especially useful when working in teams - this is the place where you can share your project code, see other people's code and integrate it into your local version of the project, and also push your changes to the remote repository.

!!! note
    Simply create this using this simple [guide](https://docs.github.com/en/get-started/quickstart/create-a-repo).

---

### Staging and Committing Code Changes

Committing is the process in which the changes are *officially* added to the Git repository.

In Git, we can consider **commits** to be checkpoints, or snapshots of your project at its current state.
In other words, we basically save the current version of our code in a commit.
We can create as many commits as we need in the commit history, and we can go back and forth between
commits to see the different revisions of our project code. 
That allows us to efficiently manage our progress and track the project as it gets developed.

Commits are usually created at logical points as we develop our project, usually after adding in 
specific contents, features or modifications (like new functionalities or bug fixes, for example).

#### Creating gitignore

Usually, I recommend creating `.gitignore` at this step, because you want to list down files 
that you do not wish git to track. 
This is important because some secret keys should never be pushed to the remote server for people to see. Furthermore, some large files should be kept in a storage as git cannot store too large files.

To ignore files that you don't want to be tracked or added to the staging area, you can create a file called `.gitignore` in your main project folder.

Inside of that file, you can list all the file and folder names that you definitely do not want to track (each ignored file and folder should go to a new line inside the **.gitignore** file).

```bash title="create .gitignore" linenums="1"
~/gaohn/git_tutorial $ touch .gitignore
```

For example, if I have a file called `secrets.py` that contains my secret key, I can add it to the `.gitignore` file.

```bash title="create .gitignore" linenums="1"
~/gaohn/git_tutorial $ touch secrets.py
~/gaohn/git_tutorial $ echo "secrets.py" >> .gitignore
```

Visit the [Atlassian's gitignore tutorial](https://www.atlassian.com/git/tutorials/saving-changes/gitignore) for more details.

#### Checking Status: `git status`

While located inside the project folder in our terminal, we can type the following command to check the status of our repository:

```bash title="git status" linenums="1"
~/gaohn/git_tutorial $ git status
```

which will return something like this:

```bash title="git status output" linenums="1"
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .gitignore
        README.md
        secrets.py

nothing added to commit but untracked files present (use "git add" to track)
```

This is a command that is very often used when working with Git.  It shows us which files have been changed, which files are tracked, etc.

We can add the untracked project files using `git add` to the **staging area** based on the information from the `git status` command.

At a later point, `git status` will report any modifications that we made to our tracked files before we decide to add them to the staging area again.

#### Adding Files to the Staging Area: `git add`

The `git add` command adds a change in the working directory to the staging area. 

It tells Git that you want to include updates to a particular file in the next commit. 

However, `git add` doesn't really affect the repository in any significant way—changes are not actually
recorded until you run git commit.

In conjunction with these commands, you'll also need git status to view the state of the working directory and the staging area.

Next we will add all our current changes to the staging area:

```bash title="git add all files in cwd" linenums="1"
~/gaohn/git_tutorial $ git add . # (1)
```

1. add all files in the current directory

You can also choose to add specific files to the staging area.

```bash title="git add specific files" linenums="1"
~/gaohn/git_tutorial $ git add <file>
```

Visit the [Atlassian's git add tutorial](https://www.atlassian.com/git/tutorials/saving-changes/git-add) for more details.

#### Making commits: `git commit`

A **commit** is a snapshot of our code at a particular time, which we are saving to the commit 
history of our repository. After adding all the files that we want to track to the staging area with
the **`git add`** command, we are ready to make a commit.

To commit the files from the staging area, we use the following command:

```bash title="git commit" linenums="1"
~/gaohn/git_tutorial $ git config --global core.editor "code --wait" # (1)
~/gaohn/git_tutorial $ git commit -a                                 # (2)
```

1. use this code if popup editor does not appear.
2. a pop up editor should appear for you to type the message.

The commit message should be a descriptive summary of the changes that you are committing to the repository. 
For the first commit, I usually type "Initial Commit" as the commit message.

After making the commit, you will see messages like the following:

```bash title="git commit output" linenums="1"
[master (root-commit) 3dccc05] Initial Commit
 3 files changed, 3 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 create mode 100644 secrets.py
```

Visit the [Atlassian's git commit tutorial](https://www.atlassian.com/git/tutorials/saving-changes/git-commit) for more details.

##### Commit History: `git log`

To see all the commits that were made for our project, you can use the following command:

```bash title="git log" linenums="1"
~/gaohn/git_tutorial $ git log
```

The logs will show details for each commit, like the author name, the generated hash for the commit, date and time of the commit, and the commit message that we provided.

To go back to a previous state of your project code that you committed, you can use the following command:

```bash title="git checkout specific commit" linenums="1"
~/gaohn/git_tutorial $ git checkout <commit_hash>
```

Replace `<commit-hash>` with the actual hash for the specific commit that you want to visit, which is listed with the `git log` command.

To go back to the latest commit (the newest version of our project code), you can type this command:

```bash
~/gaohn/git_tutorial $ git checkout master
```

##### Update/Amend Commit

To update the commit message for the latest commit, you can use the following command:

```bash title="git commit --amend" linenums="1"
~/gaohn/git_tutorial $ git commit --amend
```

This is useful if you forgot to add something to the commit message.

There are much more nuance to this and visit the [Atlassian's rewriting history 
tutorial](https://www.atlassian.com/git/tutorials/rewriting-history#git-commit--amend) for more details
on how to change committed files.


#### Comparing Changes with `git diff`

Visit [Atlassian's git diff tutorial](https://www.atlassian.com/git/tutorials/saving-changes/git-diff) for more details.

#### Stashing Changes with `git stash`

Visit [Atlassian's git stash tutorial](https://www.atlassian.com/git/tutorials/saving-changes/git-stash) for more details.

### Inspecting Repository

Visit [Atlassian's inspecting a repository tutorial](https://www.atlassian.com/git/tutorials/inspecting-a-repository) for more details.

### Undoing Changes

This section is important as it teaches you how to "undo" changes that you made to your repository.

!!! Quote "Atlassian Undoing Changes"
    In this section, we will discuss the available 'undo' Git strategies and commands. It is first important to note that Git does not have a traditional 'undo' system like those found in a word processing application. It will be beneficial to refrain from mapping Git operations to any traditional 'undo' mental model. Additionally, Git has its own nomenclature for 'undo' operations that it is best to leverage in a discussion. This nomenclature includes terms like reset, revert, checkout, clean, and more.

    A fun metaphor is to think of Git as a timeline management utility. Commits are snapshots of a point in time or points of interest along the timeline of a project's history. Additionally, multiple timelines can be managed through the use of branches. When 'undoing' in Git, you are usually moving back in time, or to another timeline where mistakes didn't happen.

    This tutorial provides all of the necessary skills to work with previous revisions of a software project. First, it shows you how to explore old commits, then it explains the difference between reverting public commits in the project history vs. resetting unpublished changes on your local machine.

Visit [Atlassian's undoing changes tutorial](https://www.atlassian.com/git/tutorials/undoing-changes) for more details.

### Rewriting History

Visit [Atlassian's rewriting history tutorial](https://www.atlassian.com/git/tutorials/rewriting-history) for more details.

#### Rewriting History with `git commit --amend`

#### Rewriting History with `git rebase`

#### Rewriting History with `git reflog`

## Collaborating

### Syncing

!!! note
    So far, everything done above is still in local repository, if we want to push our content to a
    remote repository for collaboration, we will need to harness the power of `git remote`.

#### Git Remote

To add a remote repository to our local repository, we first need to create a remote repository.

For our purpose we will be using GitHub. Follow [the steps here](https://docs.github.com/en/get-started/quickstart/create-a-repo)
to create a new repository on GitHub.

Once the remote repository is created, we will copy the URL for the remote repository. In this tutorial,
we will only focus on the HTTPS URL.

```bash title="git remote add" linenums="1"
~/gaohn/git_tutorial $ git remote add <remote name> <remote url>
```

where `<remote name>` is the name of the remote repository and `<remote url>` is the URL of the remote repository.

In our example, and typically, our remote name is `origin` and the remote URL is the HTTPS URL of the remote repository.

```bash title="git remote add" linenums="1"
~/gaohn/git_tutorial $ git remote add origin https://github.com/reigHns92/git-tutorial.git
```


!!! tip
    I often encounter permission denied error when pushing to remote when I have two github accounts
    on the same local machine, one solution is to use SSH, the other is the one listed below:

    1. Create a [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) 
       for the second account and **[tick all access](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)**. 
    
    2. Add the remote origin for the second account.
    ```bash
    git remote add origin git remote add <remote name> <remote url>
    git remote set-url <remote name> <https://[token]@github.com/[username]/[repository_name]>
    ```
    
    3. Push the changes to the second account and a prompt opens for you to key in token access.

Visit [Atlassian's git remote tutorial](https://www.atlassian.com/git/tutorials/syncing/git-remote) for more details.

##### Useful Git Remote Commands

```bash title="git remote commands" linenums="1"
~/gaohn/git_tutorial $ git remote -v                            # (1)
~/gaohn/git_tutorial $ git remote rm <remote name>              # (2)
~/gaohn/git_tutorial $ git remote rename <old name> <new name>  # (3)
```

1. List all the remote repositories that are added to our local repository.
2. Remove a remote repository from our local repository.
3. Rename a remote repository.

---

#### Git Push

!!! Quote "Atlassian Git Push"
    The `git push` command is used to upload local repository content to a remote repository. 

    Pushing is how you transfer commits from your local repository to a remote repo. 
    
    It's the counterpart to git fetch, but whereas fetching imports commits to local branches, 
    pushing exports commits to remote branches. Remote branches are configured using the git remote command. 
    
    Pushing has the potential to overwrite changes, caution should be taken when pushing. These issues are discussed below.

We can finally push our local repository to the remote repository.

```bash title="git push" linenums="1"
~/gaohn/git_tutorial $ git push <remote name> <branch name>
```

where `<remote name>` is the name of the remote repository and `<branch name>` is the name of the branch.

and in our example, we will be pushing to the `origin` remote repository and the `master` branch.

=== "Git Push"
    ```bash title="git push" linenums="1"
    ~/gaohn/git_tutorial $ git push origin master
    ```

=== "Git Push Outputs"
    ```bash title="git push output" linenums="1"
    Enumerating objects: 8, done.
    Counting objects: 100% (8/8), done.
    Delta compression using up to 12 threads
    Compressing objects: 100% (4/4), done.
    Writing objects: 100% (8/8), 614 bytes | 614.00 KiB/s, done.
    Total 8 (delta 1), reused 0 (delta 0), pack-reused 0
    remote: Resolving deltas: 100% (1/1), done.
    To https://github.com/reigHns92/git-tutorial.git
    * [new branch]      master -> master
    Branch 'master' set up to track remote branch 'master' from 'origin'.
    ```

This will now push our local repository to the remote repository.

##### Non-Fast-Forward Push

This terminology was mentioned a few times in Atlassian's Git tutorial.

Basically, if Git detects that the remote repository is ahead of the local repository, it will refuse to push.

This is common, let's see the example:

1. Prior to this section, we pushed our local repository to the remote repository `master` with commit SHA `b9d3256`.
2. Shortly after, my friend Joe synced up with the remote repository at commit SHA `b9d3256` and he updated
`README.md` and pushed with commit SHA `13e956d` to `master`.
3. Now if I run `git push -u origin master` again, Git will detect that the remote repository is ahead of the local repository and refuse to push, giving me the following errors.

    ```bash title="git push" linenums="1"
    To https://github.com/reigHns92/git-tutorial.git
    ! [rejected]        master -> master (fetch first)
    error: failed to push some refs to 'https://github.com/reigHns92/git-tutorial.git'
    hint: Updates were rejected because the remote contains work that you do
    hint: not have locally. This is usually caused by another repository pushing
    hint: to the same ref. You may want to first integrate the remote changes
    hint: (e.g., 'git pull ...') before pushing again.
    hint: See the 'Note about fast-forwards' in 'git push --help' for details
    ```
4. To resolve this, I need to `git pull origin` first and then run `git push -u origin master`.
5. However, this is an over-simplified example because I made an assumption that I did not have any local changes/developments
on at the commit `b9d3256`. If I did, then I may have to resolve merge conflicts, which we will see in future tutorials.

##### Git Push Sample Workflow

Mention about using `git rebase -i` and link to the section on it later.

#### Git Fetch

!!! Quote "Atlassian Git Fetch"
    The `git fetch` command downloads commits, files, and refs from a remote repository into your local repo. Fetching is what you do when you want to see what everybody else has been working on. It lets you see how the central history has progressed, but it doesn’t force you to actually merge the changes into your repository. Git isolates fetched content from existing local content; it has absolutely no effect on your local development work. Fetched content has to be explicitly checked out using the `git checkout` command. This makes fetching a safe way to review commits before integrating them with your local repository.

    When downloading content from a remote repo, `git pull` and `git fetch` commands are available to accomplish the task. You can consider `git fetch` the 'safe' version of the two commands. It will download the remote content but not update your local repo's working state, leaving your current work intact. git pull is the more aggressive alternative; it will download the remote content for the active local branch and immediately execute git merge to create a merge commit for the new remote content. If you have pending changes in progress this will cause conflicts and kick-off the merge conflict resolution flow.

We will add more content with examples later, for now we will see a simplified example.

Visit [Atlassian's git fetch tutorial](https://www.atlassian.com/git/tutorials/syncing/git-fetch) for more details.

##### Synchronize Origin with `git fetch`

Let's say I have two local machines A and B. I was developing on machine A.

Let's say I created a new branch `git-fetch` on another local machine B at commit SHA `13e956d`, meaning to say
this branch has a snapshot of the remote repository at commit SHA `13e956d`.

But when I get back to machine A, I want to synchronize my local repository with the remote repository.

I can call the command `git fetch origin`:

=== "Git Fetch"
    ```bash title="git fetch" linenums="1"
    ~/gaohn/git_tutorial $ git fetch origin
    ```

=== "Git Fetch Outputs"
    ```bash title="git fetch output" linenums="1"
    From https://github.com/reigHns92/git-tutorial
    * [new branch]      git-fetch  -> origin/git-fetch
    ```

This will "download" the remote repository's `git-fetch` branch to my local repository and I now have
a local branch `git-fetch` that is tracking the remote branch `git-fetch`.
  
To further understand this, let's say I made some changes to the `README.md` on branch `git-fetch`
at commit hash `13e956d` and pushed it to the remote repository from machine B at commit `f3d4f25`.
Consequently, my local repo at machine A will be one commit behind the remote repository at branch `git-fetch`.

=== "Git Fetch"
    ```bash title="git fetch" linenums="1"
    ~/gaohn/git_tutorial $ git fetch origin
    ```

=== "Git Fetch Outputs"
    ```bash title="git fetch output" linenums="1"
    remote: Enumerating objects: 5, done.
    remote: Counting objects: 100% (5/5), done.
    remote: Compressing objects: 100% (3/3), done.
    remote: Total 3 (delta 1), reused 0 (delta 0), pack-reused 0
    Unpacking objects: 100% (3/3), 742 bytes | 123.00 KiB/s, done.
    From https://github.com/reigHns92/git-tutorial
    * branch            git-fetch  -> FETCH_HEAD
    13e956d..f3d4f25  git-fetch  -> origin/git-fetch
    ```

Running `git status` will also show that the local repository at machine A is one commit behind the remote repository at branch `git-fetch`.

To see what commits have been added to the remote origin branch `git-fetch`, you can run a `git log` using `origin/git-fetch` as a filter:  

```bash title="git log" linenums="1"
~/gaohn/git_tutorial $ git log --oneline master..origin/git-fetch
```

which will a message `f3d4f25 (origin/git-fetch) git: update README`.

Finally, you can merge the changes made on `git-fetch` to your local repository at machine A:

=== "Git Merge"
    ```bash title="git merge" linenums="1"
    ~/gaohn/git_tutorial $ git merge origin/git-fetch
    ```

=== "Git Merge Outputs"
    ```bash title="git merge output" linenums="1"
    Updating 13e956d..f3d4f25
    Fast-forward
    README.md | 1 +
    1 file changed, 1 insertion(+)
    ```

---

#### Git Pull

!!! Quote "Atlassian Git Pull"
    The `git pull` command is a combination of `git fetch` and `git merge`. It downloads the remote content and merges it with your local content.


Visit [Atlassian's git pull tutorial](https://www.atlassian.com/git/tutorials/syncing/git-pull) for more details.


### Branches

A **branch** could be interpreted as an individual timeline of our project commits.

With Git, we can create many of these alternative environments (i.e. we can create different **branches**) so other versions of our project code can exist and be tracked in parallel.

That allows us to add new (experimental, unfinished, and potentially buggy) features in separate branches, without touching the '*official'* stable version of our project code (which is usually kept on the **master** branch).

When we initialize a repository and start making commits, they are saved to the **master** branch by default.

#### Creating a new branch

You can create a new branch using the following command:

```bash
git branch <new-branch-name>
```

The new branch that gets created will be the reference to the current state of your repository.

!!! note
    It's a good idea to create a **development** branch where you can work on improving your code, adding new experimental features, and similar. After development and testing these new features to make sure they don't have any bugs and that they can be used, you can merge them to the master branch.


#### Changing branches

To switch to a different branch, you use the **git checkout** command:

```bash
git checkout <branch-name>
```

With that, you switch to a different isolated timeline of your project by changing branches.

!!! note
    For example, you could be working on different features in your code and have a separate branch for each feature. When you switch to a branch, you can commit code changes which only affect that particular branch. Then, you can switch to another branch to work on a different feature, which won't be affected by the changes and commits made from the previous branch.

To create a new branch and change to it at the same time, you can use the **-b** flag:

```bash
git checkout -b <new-branch-name>
```

!!! info
    To list the branches for your project, use this command: `git branch`

To go back to the **master** branch, use this command:

```bash
git checkout master
```

#### Merging branches

You can merge branches in situations where you want to implement the code changes that you made in an individual branch to a different branch.

For example, after you fully implemented and tested a new feature in your code, you would want to merge those changes to the stable branch of your project (which is usually the default **master** branch).

To merge the changes from a different branch into your current branch, you can use this command:

```bash
git merge <branch-name>
```

You would replace `<branch-name>` with the branch that you want to integrate into your current branch.

#### Deleting a branch

To delete a branch, you can run the **git branch** command with the **-d** flag:

```bash
git branch -d <branch-name>
```

!!! info
    Read more about branching and merging [on this link](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging).

---

### Git Branches For Personal Projects Guidelines

#### Website and Blogging

Assume that I wrote a notebook named `blog.ipynb` and I want to publish it to my personal blog. Here are the general guidelines.

- Say that I put the notebook `blog.ipynb` in the `notebooks` folder in master branch.
- Then before I publish, I have a collaborator named **Joe** who is also a collaborator of my personal blog (who acts as my editor).
- I will create a new branch named `blog` and push the notebook to the branch.
- Then I will commit the notebook to the branch and create a pull request for **Joe** to review. 
    - If **Joe** approves the pull request, then I will merge the branch to the master branch using `squash and merge` or `merge commit`.
    - If **Joe** proposes some small changes, I will then go my branch `blog` and make the changes locally, and commit them with a pull request for **Joe** again. Once he approved, I will merge the branch to the master branch.
- After the merge, I will delete the branch `blog` as the updates are reflected in the master branch.

The above can be outlined in pseudo code below:

```bash
# Place the notebook `blog.ipynb` in the `notebooks` folder in master branch.
# Create a new branch named `blog` and push the notebook to the branch.
git checkout -b <new-branch-name> blog
git status # to see the status of the current branch.
# Convert the notebook to markdown if needed.
jupyter nbconvert --to markdown mynotebook.ipynb
# Commit the notebook to the branch.
git add .
git commit -a
git push origin branch_name -u
# Go to the pull request link, and select reviewer for review.
# If no changes proposed, merge the branch to master.
# If there are changes proposed, edit the codes in the branch locally and push the changes to the branch for review again. After that go to the same link and click on re-review.
```

After review and approved by the reviewer, you have 3 merge options.

- Squash and Merge: Take all the commits during the review and merge them into one commit. The only downside is you cannot go back to one unique commit but you will see all commit messages tho and changes.
- Create a merge commit: Create a merge commit with the changes from the review. You can use this if you want `git` to track every single commit (messages).

The merge will mean it is updated in master branch also. Then delete the branch if not in use.

---

## Contributing to Open Source Projects, A Generic Sample Workflow

This workflow described in this section is a generic workflow, for other detailed models of workflows,
refer to [Atlassian's Git Workflows](https://www.atlassian.com/git/tutorials/comparing-workflows) for more details.

### Fork the Repository

We will first start off by going to the repository you are interested in and click on the **Fork** 
button on the top right.

I will use a [public repository](https://github.com/ghnreigns/git-sample-workflow) named **git-sample-workflow** from my other account.

This will replicate his entire repository into my own GitHub account. 
Remember to uncheck the box **Copy the `main` branch only**
if you decide to work on the forked repository's branches as well.

### Clone your Forked Repository

I have a fork of the **test** repository on my GitHub server, but to start developing, I will have to 
clone this forked repository locally on my computer.

Depending on HTTPS or SSH key, one might copy different URLs for the forked repository.


=== "Git Clone Forked Repo"

    ```bash title="git clone forked repo"
    ~/gaohn $ git clone https://github.com/reigHns92/git-sample-workflow.git
    ```

=== "Outputs"

    ```
    Cloning into 'git-sample-workflow'...
    remote: Enumerating objects: 6, done.
    remote: Counting objects: 100% (6/6), done.
    remote: Compressing objects: 100% (3/3), done.
    remote: Total 6 (delta 0), reused 0 (delta 0), pack-reused 0
    Receiving objects: 100% (6/6), done.
    ```

For a more detailed step-by-step guide, [GitHub Docs's Quickstart: Fork a repo](https://
docs.github.com/en/get-started/quickstart/fork-a-repo)
provides a templated method to do fork and clone.

### Sync Forked Repository with Upstream

Before we go on, we define [two key terms](https://stackoverflow.com/questions/9257533/what-is-the-difference
-between-origin-and-upstream-on-github#:~:text=upstream%20generally%20refers%20to%20the,
the%20original%20repo%20of%20GitHub) in the context of [GitHub Forks](https://help.github.com/articles/fork-a-repo/):

- `origin`: this refers to one's own repository (i.e. the forked and cloned repository on your personal account);
- `upstream`: generally refers to the original repository that you have forked.

In general, it is also useful to understand [upstreams and downstreams](https://stackoverflow.com/que
stions/2739376/definition-of-downstream-and-upstream/2749166#2749166).

Back to where we left off, you can continue to develop on the forked repository but the
**original repository** you forked from (the `upstream`) will not automatically sync with
your `origin`. In other words, if the `upstream` repository made 10 new commits, your `origin`
will have no information on those 10 new commits.

To be able to sync changes, we will have to first configure a remote for the fork, and fetch from upstream to sync.

#### Configuring a Remote for a Forked Repository

The steps are referenced from [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with
-pull-requests/working-with-forks/configuring-a-remote-for-a-fork). They have included steps for 
different OS. We will reproduce the steps on macOS.

1. Open the terminal;

2. A handy command to check what remote repository your forked repo has. Note that one can have many
   remotes! 

    ```bash title="list out all remotes" linenums="1"
    ~/gaohn                     $ cd git-sample-workflow
    ~/gaohn/git-sample-workflow $ git remote -v
    
    > origin	https://github.com/reigHns92/git-sample-workflow.git (fetch)
    > origin	https://github.com/reigHns92/git-sample-workflow.git (push)
    ```

3. Specify a new remote *upstream* repository that will be synced with the fork.
   To do so, go to the original [repository](https://github.com/ghnreigns/git-sample-workflow.git) (the one you forked from) and copy the
   URLs (HTTPS or SSH) to add to remote (upstream).
   
    ```bash title="configuring remote for forked repo" linenums="1"
    ~/gaohn/git-sample-workflow $ git remote add upstream https://github.com/ghnreigns/git-sample-workflow.git

    > origin	https://github.com/reigHns92/git-sample-workflow.git (fetch)
    > origin	https://github.com/reigHns92/git-sample-workflow.git (push)
    ```

    You can verify the new upstream repository is added by checking `git remove -v` again.

    ```bash title="list out all remotes" linenums="1"
    ~/gaohn/git-sample-workflow $ git remote -v
    
    > origin	https://github.com/reigHns92/git-sample-workflow.git (fetch)
    > origin	https://github.com/reigHns92/git-sample-workflow.git (push)
    > upstream	https://github.com/ghnreigns/git-sample-workflow.git (fetch)
    > upstream	https://github.com/ghnreigns/git-sample-workflow.git (push)
    ```

#### Syncing the Forked Repository with Upstream 

Now I will use my other account to make a new commit [0780e45
](https://github.com/ghnreigns/github-test/commit/0780e45e0efba76881477af91474a05c1da9ae4e) on the repository's main branch (i.e the upstream) and now
my forked repository will be 1 commit behind.

<figure markdown>
  ![Image title](../../assets/software_engineering/git/main_1.png){ width="600" }
  <figcaption>Image caption</figcaption>
</figure>

Note that the `dev` branch that we forked remains the same as it is still in sync with the upstream's `dev` branch, hence the 
image below shows that as well:

<figure markdown>
  ![Image title](../../assets/software_engineering/git/dev_1.png){ width="600" }
  <figcaption>Image caption</figcaption>
</figure>

To keep my repository up to date, we will refer to [GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-
requests/working-with-forks/syncing-a-fork#syncing-a-fork-branch-from-the-command-line)
section on syncing a fork branch from the command line.

1. In the same terminal, we fetch all branches and commites from the upstream repository.

    ```bash title="fetching from upstream repo" linenums="1"
    ~/gaohn/github-test $ git fetch upstream
    > remote: Enumerating objects: 5, done.
    > remote: Counting objects: 100% (5/5), done.
    > remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
    > Unpacking objects: 100% (3/3), 653 bytes | 217.00 KiB/s, done.
    > From https://github.com/ghnreigns/github-test
    > * [new branch]      dev        -> upstream/dev
    > * [new branch]      main       -> upstream/main
    ```

2. Switch to your fork's main branch (or the branch you want the changes to be synced and merged).
       
3. Merge the changes from the upstream default branch - in this case, `upstream/main` - into your local default branch. 
   This brings your fork's default branch into sync with the upstream repository, without losing your local changes.

    ```bash title="merge with upstream" linenums="1"
    ~/gaohn/github-test $ git merge upstream/main
    > Updating 3faaf86..0b69046
    > Fast-forward
    > hello.py | 8 ++++++++
    > 1 file changed, 8 insertions(+)
    > create mode 100644 hello.py
    ```

After this step, the origin (forked) repo of mine will be in sync with the upstream as can be verified by
the message "This branch is up to date with ghnreigns/github-test:main." on your main branch github.

!!! warning
    Notice that the `dev` branch does not automatically appear in your local git (i.e. calling `git checkout dev` does not work).
    The `dev` branch exists in both `origin` and `upstream` as can be seen by `git branch -a`.

    There are [two ways](https://www.freecodecamp.org/news/git-checkout-remote-branch-tutorial/) to "fetch" the `dev` repo to your local:

    - `git checkout -b dev upstream/dev` will create a branch named `dev` that pulls the information from the `upstream/dev` branch;
    - Alternatively, you can `git checkout -b dev origin/main` which essentially fetches the `dev` branch on your forked repo and then
    we can do `git merge upstream/dev` to be in sync.


#### Creating Pull Request

Now on my local `dev` branch, I made some changes and committed to [c5db9a0](https://github.com/reigHns92/github-test/commit/c5db9a0afd0b46c6851d75e2e599356dae3cc6e8)
using `git push -u origin dev`.

I now want to create a pull request for the upstream repository owner to see my changes, and to merge into his repo if necessary.

We first go to our origin (forked) repo and see that there is a new popup on "compare and pull request".
We click it and the following interface will appear, that is for you to write a message to the upstream author.

<figure markdown>
  ![Image title](../../assets/software_engineering/git/pull_request_1.png){ width="600" }
  <figcaption>Image caption</figcaption>
</figure>

Once you created pull request, you can wait for the upstream author to decide and reply.



## Commit Courtesy

!!! tip
    Files that are changed in sync should be committed together. As an example, you made changes to three files, a, b and c, if a and b are related and c is unrelated to both of them, then it is logical to do the following:
    ```bash
    git add a b
    git commit -a "Commit related files"
    git add c
    git commit -a "Commit c file"
    ```

and https://github.com/aisingapore/PeekingDuck/blob/main/CONTRIBUTING.md



## GIT Readme Profile

Instructions [here](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/managing-your-profile-readme) and [template](https://github.com/khuyentran1401/khuyentran1401/blob/master/README.md).