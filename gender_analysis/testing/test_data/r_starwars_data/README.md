# r/StarWars Top Comment Corpus

This is a collection of the 8 top posts from [r/StarWars](https://www.reddit.com/r/starwars), along with a selection of the top comments from each of those posts (for a total of 9,993 comment posts). These comments are stored _without_ connection to their parent posts.

## Metadata File
The metadata for the corpus can be found in the top level directory at `metadata.csv` and contains the following fields:

- `filename`: the name of the file in `posts/` that the row is referring to. Note that this file contains the text of the post/comment, and therefore may be empty if the post is a non-text post.
- `is_comment`: whether the post is a post (False) or a comment (True).
- `author_fullname`: the hidden author id that Reddit assigns to users
- `created_utc`: The time (in seconds since the epoch) that the post was created in the UTC timezone
- `downs`: The number of downvotes that the post/comment received
- `gilded`: The number of "gold" awards that the post/comment was awarded
- `likes`: The number of likes the post has received (note that most posts/comments do not have this field)
- `name`: The unique post name that is assigned by Reddit
- `num_comments`: The number of comments that a post/comment has underneath it. 
- `score`: The score that Reddit assigns the post. This is equivalent to the post/comment's karma, and as such is slightly obfuscated by Reddit. A very general relation that can describe the number is `ups - downs`.
- `title`: The post's title. Note that comments do not have this field.
- `total_awards_received`: The total number of Reddit awards that have been allocated to the post/comment. This number is greater than or equal to `gilded` based upon the number of non-gold awards the post received.
- `ups`: The number of upvotes the post has received.
- `upvote_ratio`: Roughly equivalent to the ratio `ups` / `downs`. This is not applicable to comments.
- `view_count`: The number of views the post has received. Not applicable to comments.

## Post/Comment Files
In the `post` directory, there is a collection of text files, where each file represents one comment/post. The filenames are based on the post or comment's unique Reddit ID, and the metadata for each file can be found by searching for the file in `metadata.csv` file in the top level directory. Files exist regardless of the post or comment contents. The consequence of this is that non-text posts (i.e. link or image posts) are represented as empty files, and comments that contain only whitespace are still included.