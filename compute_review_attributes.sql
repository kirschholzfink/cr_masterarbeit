/*
 Add "Patch Set Count" column to table of changes.
 */
alter table t_change
    add column ch_patchSetCount int;

/*
 Insert number of patch sets for each review.
*/
update t_change ch0, (select ch1.id                   as changeId,
                             max(rev.rev_patchSetNum) as overallPatchCount
                      from t_change ch1
                               join t_revision rev
                      where rev.rev_changeId = ch1.id
                      group by ch1.id) as maxPatchesByChangeId
set ch0.ch_patchSetCount = maxPatchesByChangeId.overallPatchCount
where ch0.id = maxPatchesByChangeId.changeId;

/*
 Add "affectedFilesCount" column to table of changes.
 */
alter table t_change
    add column ch_affectedFilesCount int;

/*
 Insert overall number of files that were affected by a change.
 */
update t_change ch, (select rev.rev_changeId        as changeId,
                            count(distinct file.id) as fileCount
                     from t_revision rev
                              join t_file file on f_revisionId = rev.id
                     group by rev.rev_changeId) as fileCountByChangeId
set ch.ch_affectedFilesCount = fileCountByChangeId.fileCount
where ch.id = fileCountByChangeId.changeId;

/*
 Add "churnSize" column to table of changes.
 */
alter table t_change
    add column ch_churnSize int;

/*
 Insert overall churn size associated with a change.
 */
update t_change ch, (select rev.rev_changeId           as changeId,
                            (sum(file.f_linesInserted) +
                             sum(file.f_linesDeleted)) as totalChurnSize
                     from t_revision rev
                              join t_file file on rev.id = file.f_revisionId
                     group by changeId) as churnSizeByChangeId
set ch.ch_churnSize = churnSizeByChangeId.totalChurnSize
where ch.id = churnSizeByChangeId.changeId;

/*
 Add "initialResponseTimeInHours" column to table of changes.
 */
alter table t_change
    add column ch_initialResponseTimeInHours int;

/*
 Add "initialCommentId" column to table of changes.
 */
alter table t_change
    add column ch_initialCommentId int unique;

/*
 Insert initial response time to a review request (change) in hours.
 Insert id of initial comment.

 Initial response time is computed as the difference between
 the timestamp of when the pull request (change) was created
 and the earliest timestamp of all comments associated to it.

 Only those comments that were written by a reviewer
 (as opposed to comments written by the author themselves)
 are considered as potential initial responses.
 See subquery alias "reviewerComments".
 */
update t_change ch0, (select responseTimeByChangeId.changeId,
                             responseTimeByChangeId.initialResponseTime as initialResponseTime,
                             comm0.id                                   as initialCommentId
                      from t_history comm0
                               join (select ch1.id                          as changeId,
                                            ch_createdTime,
                                            min(reviewerComments.timeStamp) as firstCommentTimeStamp,
                                            datediff(
                                                    min(reviewerComments.timeStamp),
                                                    ch1.ch_createdTime)     as initialResponseTime
                                     from t_change ch1
                                              join (select comm1.hist_changeId    as changeId,
                                                           comm1.hist_createdTime as timeStamp
                                                    from t_history comm1
                                                             join t_change ch2 on ch2.id = comm1.hist_changeId
                                                    where ch2.ch_authorAccountId !=
                                                          comm1.hist_authorAccountId) as reviewerComments
                                                   on ch1.id = reviewerComments.changeId
                                     group by reviewerComments.changeId) as responseTimeByChangeId
                                    on responseTimeByChangeId.changeId =
                                       comm0.hist_changeId and
                                        /*This works because the timestamp
                                          of a comment hist_createdTime
                                          is unique within a given change.*/
                                       responseTimeByChangeId.firstCommentTimeStamp =
                                       comm0.hist_createdTime) as responseTimesByChangeId
set ch0.ch_initialResponseTimeInHours = responseTimesByChangeId.initialResponseTime,
    ch0.ch_initialCommentId           = responseTimesByChangeId.initialCommentId
where ch0.id = responseTimesByChangeId.changeId;

/*
 Add "authorial sentiment" column to table of changes.
 */

alter table t_change
    add column ch_authorialSentiment varchar(30);

/*
 Insert sentiment 'negative' or 'non-negative'
 into column "ch_authorialSentiment".
 Cell is set to 'negative' if at least one comment that is made
 by the CR request author themselves was classified as 'negative'.
 Negative comments made by reviewers are discarded.
 If no negative comments made by the author are present
 in the comments surrounding a CR request, cell is set to 'non-negative'.
 */

update t_change ch0, (select hist_changeId as changeId
                      from t_history) as comments

set ch0.ch_authorialSentiment =
        IF(ch0.id in (select *
                      from (select ch1.id as changeId
                            from t_change ch1
                                     join t_history comm on ch1.id
                                                        = comm.hist_changeId
                            where ch1.ch_authorAccountId = hist_authorAccountId
                              and comm.id in
                                  (select id
                                   from t_history
                                   where comm.sentiment like 'negative'))
                          as negativeCommentsByAuthors),
           'negative', 'non-negative')

where ch0.id = comments.changeId;

/*
 Add "reviewer sentiment" column to table of changes.
 */

alter table t_change
    add column ch_reviewerSentiment varchar(30);

/*
 Insert sentiment 'negative' or 'non-negative'
 into column "ch_reviewerSentiment".
 Cell is set to 'negative' if at least one comment that is made
 by the CR request reviewer themselves was classified as 'negative'.
 Negative comments made by the code author are discarded.
 If no negative comments made by the reviewer are present
 in the comments surrounding a CR request, cell is set to 'non-negative'.
 */

update t_change ch0, (select hist_changeId as changeId
                      from t_history) as comments

set ch0.ch_reviewerSentiment =
        IF(ch0.id in (select *
                      from (select ch1.id as changeId
                            from t_change ch1
                                     join t_history comm on ch1.id
                                                        = comm.hist_changeId
                            where ch1.ch_authorAccountId != hist_authorAccountId
                              and comm.id in
                                  (select id
                                   from t_history
                                   where comm.sentiment like 'negative'))
                          as negativeCommentsByReviewers),
           'negative', 'non-negative')

where ch0.id = comments.changeId;

/*
 How many negative comments
 were made by the authors of CR requests themselves?
 */

select count(*)
from t_change ch
         join t_history comm on ch.id = comm.hist_changeId
where ch_authorAccountId = hist_authorAccountId
  and comm.id in
      (select id from t_history where comm.sentiment like 'negative');

/*
 How many negative comments were made by the reviewers of CR requests?
 */

select count(*)
from t_change ch
         join t_history comm on ch.id = comm.hist_changeId
where ch_authorAccountId != hist_authorAccountId
  and comm.id in
      (select id from t_history where comm.sentiment like 'negative');
