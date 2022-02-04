/*
 Delete all reviews in t_change that have not yet been closed.
 */
delete
from t_change
where ch_status LIKE 'new';

/*
 Delete all comments that were automatically generated by bots.
 This is equivalent to the following query:
    delete
    from t_history
    WHERE hist_message LIKE '%Build Started%'
       or hist_message like '%Build Successful%'
       or hist_message like '%Build Failed%'
       or hist_message like
        '%Change has been successfully cherry-picked as%'
       or hist_message like '%Change has been successfully pushed.%'
       or hist_message like '%was rebased%'
       or hist_message like '%Cherry Picked from branch master.%'
       or hist_message like '%Uploaded patch set%'
       or hist_message like '%Build failure%'
       or hist_message like '%Merge Failed%'
       or hist_message like '%could not be merged due to a path conflict.%'
       or hist_message like
        '%Change has been successfully merged into the git repository%';
 */
delete
from t_history
where hist_authorAccountId like '';


/*
 Delete all reviews in t_change
 that have not been commented on by both the author and a reviewer.
 */
delete
from t_change
where id not in (select *
                 from (select comm.hist_changeId
                       from t_change ch
                                join t_history comm
                                    on ch.id = comm.hist_changeId
                       where ch.ch_authorAccountId !=
                             comm.hist_authorAccountId)
                     as commentsByReviewers)
   or id not in (select *
                 from (select comm.hist_changeid
                       from t_change ch
                                join t_history comm
                                    on ch.id = comm.hist_changeId
                       where ch.ch_authorAccountId =
                             comm.hist_authorAccountId)
                     as commentsByAuthor);

/*
 Delete comments in t_history that are associated
 with changes that were deleted.
 (This is required as there is no 'on delete/update cascade' action
 defined in db schema.)
 */
delete
from t_history
where hist_changeId not in (select id from t_change);
/*
 Delete all revisions in t_revision that are associated
 with changes that were deleted.
 (This is required as there is no 'on delete/update cascade' action
 defined in db schema.)
 */
delete
from t_revision
where rev_changeId not in (select id from t_change);
/*
 Delete all files in t_file that are associated
 with revisions that were deleted.
 (This is required as there is no 'on delete/update cascade' action
 defined in db schema.)
 */
delete
from t_file
where f_revisionId not in (select id from t_revision);
